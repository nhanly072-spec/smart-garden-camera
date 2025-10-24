/*
 * ESP32-S3 + OV3660 + Edge Impulse FIRE_DETECTION
 * -----------------------------------------------
 * - Board: ESP32-S3 Dev Module (WROOM-N16R8)
 * - Camera: OV3660
 * - PSRAM: Enabled
 */

#include <FIRE_DETECTION_inferencing.h>
#include "edge-impulse-sdk/dsp/image/image.hpp"
#include "esp_camera.h"

// ======================== CAMERA PIN CONFIG (ESP32-S3 + OV3660) ========================
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM     4
#define SIOC_GPIO_NUM     5
#define Y9_GPIO_NUM       16
#define Y8_GPIO_NUM       17
#define Y7_GPIO_NUM       18
#define Y6_GPIO_NUM       12
#define Y5_GPIO_NUM       10
#define Y4_GPIO_NUM       8
#define Y3_GPIO_NUM       9
#define Y2_GPIO_NUM       11
#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     7
#define PCLK_GPIO_NUM     13
// ======================================================================================

#define EI_CAMERA_RAW_FRAME_BUFFER_COLS  320
#define EI_CAMERA_RAW_FRAME_BUFFER_ROWS  240
#define EI_CAMERA_FRAME_BYTE_SIZE        3

static bool debug_nn = false;
static bool is_initialised = false;
static uint8_t *snapshot_buf = NULL;

// ====================== CAMERA CONFIG ======================
static camera_config_t camera_config = {
    .pin_pwdn       = PWDN_GPIO_NUM,
    .pin_reset      = RESET_GPIO_NUM,
    .pin_xclk       = XCLK_GPIO_NUM,
    .pin_sscb_sda   = SIOD_GPIO_NUM,
    .pin_sscb_scl   = SIOC_GPIO_NUM,
    .pin_d7         = Y9_GPIO_NUM,
    .pin_d6         = Y8_GPIO_NUM,
    .pin_d5         = Y7_GPIO_NUM,
    .pin_d4         = Y6_GPIO_NUM,
    .pin_d3         = Y5_GPIO_NUM,
    .pin_d2         = Y4_GPIO_NUM,
    .pin_d1         = Y3_GPIO_NUM,
    .pin_d0         = Y2_GPIO_NUM,
    .pin_vsync      = VSYNC_GPIO_NUM,
    .pin_href       = HREF_GPIO_NUM,
    .pin_pclk       = PCLK_GPIO_NUM,
    .xclk_freq_hz   = 20000000,
    .ledc_timer     = LEDC_TIMER_0,
    .ledc_channel   = LEDC_CHANNEL_0,
    .pixel_format   = PIXFORMAT_JPEG,  // OV3660 ổn định nhất với JPEG
    .frame_size     = FRAMESIZE_QVGA,  // 320x240
    .jpeg_quality   = 12,
    .fb_count       = 1,
    .fb_location    = CAMERA_FB_IN_PSRAM,
    .grab_mode      = CAMERA_GRAB_WHEN_EMPTY
};

// ====================== FUNCTION DECLARATIONS ======================
bool ei_camera_init(void);
void ei_camera_deinit(void);
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf);
static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr);

// ====================== SETUP ======================
void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("\n🔥 Edge Impulse Fire Detection - ESP32-S3 + OV3660\n");

  if (!ei_camera_init()) {
    Serial.println("❌ Camera init failed!");
    while (true);
  }
  Serial.println("✅ Camera initialized successfully");

  ei_printf("\nStarting continuous inference in 2 seconds...\n");
  ei_sleep(2000);
}

// ====================== LOOP ======================
void loop() {
  if (ei_sleep(5) != EI_IMPULSE_OK) return;

  if (!snapshot_buf) {
    snapshot_buf = (uint8_t*)ps_malloc(
      EI_CAMERA_RAW_FRAME_BUFFER_COLS *
      EI_CAMERA_RAW_FRAME_BUFFER_ROWS *
      EI_CAMERA_FRAME_BYTE_SIZE
    );
    if (!snapshot_buf) {
      ei_printf("❌ Failed to allocate PSRAM buffer!\n");
      return;
    }
  }

  if (!ei_camera_capture(EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT, snapshot_buf)) {
    ei_printf("❌ Image capture failed!\n");
    return;
  }

  ei::signal_t signal;
  signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * 3;
  signal.get_data = &ei_camera_get_data;

  ei_impulse_result_t result = { 0 };
  EI_IMPULSE_ERROR err = run_classifier(&signal, &result, debug_nn);
  if (err != EI_IMPULSE_OK) {
    ei_printf("Classifier error (%d)\n", err);
    return;
  }

  ei_printf("Predictions (DSP: %d ms, Classification: %d ms)\n",
            result.timing.dsp, result.timing.classification);

#if EI_CLASSIFIER_OBJECT_DETECTION == 1
  for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
    auto bb = result.bounding_boxes[i];
    if (bb.value == 0) continue;
    ei_printf("  %s (%f) [x:%u y:%u w:%u h:%u]\n",
              bb.label, bb.value, bb.x, bb.y, bb.width, bb.height);
  }
#else
  for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    ei_printf("  %s: %.5f\n",
              ei_classifier_inferencing_categories[i],
              result.classification[i].value);
  }
#endif

#if EI_CLASSIFIER_HAS_ANOMALY
  ei_printf("Anomaly: %.3f\n", result.anomaly);
#endif
}

// ====================== CAMERA INIT ======================
bool ei_camera_init(void) {
  if (is_initialised) return true;

  esp_err_t err = esp_camera_init(&camera_config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed 0x%x\n", err);
    return false;
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);
    s->set_hmirror(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, 0);
  }

  is_initialised = true;
  return true;
}

void ei_camera_deinit(void) {
  if (!is_initialised) return;
  esp_camera_deinit();
  is_initialised = false;
}

// ====================== CAPTURE ======================
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf) {
  if (!is_initialised) return false;

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    ei_printf("Capture failed\n");
    return false;
  }

  bool converted = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, out_buf);
  esp_camera_fb_return(fb);

  if (!converted) {
    ei_printf("Convert JPEG→RGB888 failed\n");
    return false;
  }

  if (img_width != EI_CAMERA_RAW_FRAME_BUFFER_COLS ||
      img_height != EI_CAMERA_RAW_FRAME_BUFFER_ROWS) {
    ei::image::processing::crop_and_interpolate_rgb888(
      out_buf, EI_CAMERA_RAW_FRAME_BUFFER_COLS, EI_CAMERA_RAW_FRAME_BUFFER_ROWS,
      out_buf, img_width, img_height);
  }

  return true;
}

// ====================== DATA ACCESS ======================
static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr) {
  size_t pixel_ix = offset * 3;
  for (size_t i = 0; i < length; i++) {
    uint8_t r = snapshot_buf[pixel_ix];
    uint8_t g = snapshot_buf[pixel_ix + 1];
    uint8_t b = snapshot_buf[pixel_ix + 2];
    out_ptr[i] = (r + g + b) / (3.0f * 255.0f); // normalize to [0,1]
    pixel_ix += 3;
  }
  return 0;
}
**
