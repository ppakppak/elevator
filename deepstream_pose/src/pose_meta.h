/*
 * Human Pose Metadata Header
 * 포즈 추정 결과를 저장하기 위한 사용자 정의 메타데이터 구조체
 */

#ifndef __POSE_META_H__
#define __POSE_META_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 최대 상수
#define POSE_MAX_KEYPOINTS 18
#define POSE_MAX_LINKS 20
#define POSE_MAX_HUMANS 100

// 키포인트 인덱스 (COCO 기준)
typedef enum {
    KEYPOINT_NOSE = 0,
    KEYPOINT_LEFT_EYE,
    KEYPOINT_RIGHT_EYE,
    KEYPOINT_LEFT_EAR,
    KEYPOINT_RIGHT_EAR,
    KEYPOINT_LEFT_SHOULDER,
    KEYPOINT_RIGHT_SHOULDER,
    KEYPOINT_LEFT_ELBOW,
    KEYPOINT_RIGHT_ELBOW,
    KEYPOINT_LEFT_WRIST,
    KEYPOINT_RIGHT_WRIST,
    KEYPOINT_LEFT_HIP,
    KEYPOINT_RIGHT_HIP,
    KEYPOINT_LEFT_KNEE,
    KEYPOINT_RIGHT_KNEE,
    KEYPOINT_LEFT_ANKLE,
    KEYPOINT_RIGHT_ANKLE,
    KEYPOINT_NECK,
    KEYPOINT_COUNT
} PoseKeypointIndex;

// 키포인트 구조체
typedef struct {
    float x;          // 정규화된 x 좌표 (0~1)
    float y;          // 정규화된 y 좌표 (0~1)
    float confidence; // 신뢰도 (0~1)
    int valid;        // 유효 여부
} PoseKeypoint;

// 단일 인간 포즈
typedef struct {
    PoseKeypoint keypoints[POSE_MAX_KEYPOINTS];
    float score;      // 전체 포즈 점수
    int num_valid;    // 유효한 키포인트 수

    // 바운딩 박스 (픽셀 좌표)
    float bbox_left;
    float bbox_top;
    float bbox_width;
    float bbox_height;

    // 이벤트 감지 결과
    int is_fallen;           // 쓰러짐 감지
    float fall_confidence;   // 쓰러짐 신뢰도
} HumanPose;

// 프레임별 포즈 메타데이터
typedef struct {
    uint32_t frame_num;
    uint32_t source_id;

    int num_humans;
    HumanPose humans[POSE_MAX_HUMANS];

    // 싸움 감지 (다중 인물)
    int is_fighting;
    float fight_confidence;
} PoseFrameMeta;

// 스켈레톤 연결 정의
static const int POSE_SKELETON[POSE_MAX_LINKS][2] = {
    {KEYPOINT_LEFT_ANKLE, KEYPOINT_LEFT_KNEE},      // 0
    {KEYPOINT_LEFT_KNEE, KEYPOINT_LEFT_HIP},        // 1
    {KEYPOINT_RIGHT_ANKLE, KEYPOINT_RIGHT_KNEE},    // 2
    {KEYPOINT_RIGHT_KNEE, KEYPOINT_RIGHT_HIP},      // 3
    {KEYPOINT_LEFT_HIP, KEYPOINT_RIGHT_HIP},        // 4
    {KEYPOINT_LEFT_SHOULDER, KEYPOINT_LEFT_HIP},    // 5
    {KEYPOINT_RIGHT_SHOULDER, KEYPOINT_RIGHT_HIP},  // 6
    {KEYPOINT_LEFT_SHOULDER, KEYPOINT_RIGHT_SHOULDER}, // 7
    {KEYPOINT_LEFT_SHOULDER, KEYPOINT_LEFT_ELBOW},  // 8
    {KEYPOINT_RIGHT_SHOULDER, KEYPOINT_RIGHT_ELBOW},// 9
    {KEYPOINT_LEFT_ELBOW, KEYPOINT_LEFT_WRIST},     // 10
    {KEYPOINT_RIGHT_ELBOW, KEYPOINT_RIGHT_WRIST},   // 11
    {KEYPOINT_LEFT_EYE, KEYPOINT_RIGHT_EYE},        // 12
    {KEYPOINT_NOSE, KEYPOINT_LEFT_EYE},             // 13
    {KEYPOINT_NOSE, KEYPOINT_RIGHT_EYE},            // 14
    {KEYPOINT_LEFT_EYE, KEYPOINT_LEFT_EAR},         // 15
    {KEYPOINT_RIGHT_EYE, KEYPOINT_RIGHT_EAR},       // 16
    {KEYPOINT_NOSE, KEYPOINT_NECK},                 // 17
    {KEYPOINT_NECK, KEYPOINT_LEFT_SHOULDER},        // 18
    {KEYPOINT_NECK, KEYPOINT_RIGHT_SHOULDER}        // 19
};

// 키포인트 이름
static const char* POSE_KEYPOINT_NAMES[POSE_MAX_KEYPOINTS] = {
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"
};

#ifdef __cplusplus
}
#endif

#endif /* __POSE_META_H__ */
