/*
 * DeepStream Custom Parser for Human Pose Estimation
 * trt_pose 모델의 출력 (heatmap, PAF)을 처리하여 포즈 추출
 *
 * 이 파서는 Bottom-up 방식으로 동작합니다:
 * 1. Heatmap에서 키포인트 후보 추출 (NMS)
 * 2. PAF를 이용하여 키포인트 연결
 * 3. Hungarian Algorithm으로 최적 매칭
 */

#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <queue>
#include <map>

#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer.h"

// 포즈 추정 상수
#define NUM_KEYPOINTS 18
#define NUM_LINKS 20
#define NMS_WINDOW_SIZE 5
#define HEATMAP_THRESHOLD 0.1f
#define PAF_SCORE_THRESHOLD 0.1f
#define PAF_SAMPLE_POINTS 10
#define MAX_HUMANS 100

// 키포인트 구조체
struct Keypoint {
    float x;
    float y;
    float score;
    int id;
};

// 연결 구조체
struct Connection {
    int part_idx1;
    int part_idx2;
    float score;
};

// 인간 포즈 구조체
struct HumanPose {
    Keypoint keypoints[NUM_KEYPOINTS];
    float score;
    int num_parts;
};

// 스켈레톤 연결 정의 (COCO 기준)
static const int SKELETON[NUM_LINKS][2] = {
    {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
    {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8},
    {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2},
    {1, 3}, {2, 4}, {0, 17}, {17, 5}, {17, 6}
};

// PAF 인덱스 매핑 (각 연결에 대한 PAF 채널)
static const int PAF_INDEX[NUM_LINKS][2] = {
    {0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9},
    {10, 11}, {12, 13}, {14, 15}, {16, 17}, {18, 19},
    {20, 21}, {22, 23}, {24, 25}, {26, 27}, {28, 29},
    {30, 31}, {32, 33}, {34, 35}, {36, 37}, {38, 39}
};


/**
 * Non-Maximum Suppression (NMS) for heatmap peaks
 */
static void nms_heatmap(
    const float* heatmap,
    int height,
    int width,
    int part_idx,
    float threshold,
    std::vector<Keypoint>& keypoints)
{
    int window = NMS_WINDOW_SIZE;
    int half_window = window / 2;

    for (int y = half_window; y < height - half_window; y++) {
        for (int x = half_window; x < width - half_window; x++) {
            float val = heatmap[y * width + x];

            if (val < threshold) continue;

            bool is_peak = true;
            for (int dy = -half_window; dy <= half_window && is_peak; dy++) {
                for (int dx = -half_window; dx <= half_window; dx++) {
                    if (dy == 0 && dx == 0) continue;
                    if (heatmap[(y + dy) * width + (x + dx)] >= val) {
                        is_peak = false;
                        break;
                    }
                }
            }

            if (is_peak) {
                Keypoint kp;
                kp.x = static_cast<float>(x) / width;
                kp.y = static_cast<float>(y) / height;
                kp.score = val;
                kp.id = static_cast<int>(keypoints.size());
                keypoints.push_back(kp);
            }
        }
    }
}


/**
 * PAF score 계산 (두 키포인트 사이)
 */
static float compute_paf_score(
    const float* paf_x,
    const float* paf_y,
    int height,
    int width,
    const Keypoint& kp1,
    const Keypoint& kp2)
{
    float dx = kp2.x - kp1.x;
    float dy = kp2.y - kp1.y;
    float dist = std::sqrt(dx * dx + dy * dy);

    if (dist < 1e-6) return 0.0f;

    float vx = dx / dist;
    float vy = dy / dist;

    float score = 0.0f;
    int valid_count = 0;

    for (int i = 0; i < PAF_SAMPLE_POINTS; i++) {
        float t = static_cast<float>(i) / (PAF_SAMPLE_POINTS - 1);
        float px = kp1.x + t * dx;
        float py = kp1.y + t * dy;

        int ix = static_cast<int>(px * width);
        int iy = static_cast<int>(py * height);

        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
            int idx = iy * width + ix;
            float paf_vx = paf_x[idx];
            float paf_vy = paf_y[idx];

            // 내적 (PAF 벡터와 연결 방향의 일치도)
            score += vx * paf_vx + vy * paf_vy;
            valid_count++;
        }
    }

    if (valid_count > 0) {
        score /= valid_count;
    }

    return score;
}


/**
 * 탐욕적 이분 매칭 (Greedy Bipartite Matching)
 */
static void greedy_matching(
    const std::vector<Keypoint>& keypoints1,
    const std::vector<Keypoint>& keypoints2,
    const std::vector<std::vector<float>>& score_matrix,
    float threshold,
    std::vector<Connection>& connections)
{
    std::vector<bool> used1(keypoints1.size(), false);
    std::vector<bool> used2(keypoints2.size(), false);

    // 스코어 기준 내림차순 정렬을 위한 인덱스 쌍
    std::vector<std::tuple<float, int, int>> scores;
    for (size_t i = 0; i < keypoints1.size(); i++) {
        for (size_t j = 0; j < keypoints2.size(); j++) {
            if (score_matrix[i][j] > threshold) {
                scores.push_back(std::make_tuple(score_matrix[i][j], i, j));
            }
        }
    }

    std::sort(scores.begin(), scores.end(),
        [](const auto& a, const auto& b) {
            return std::get<0>(a) > std::get<0>(b);
        });

    for (const auto& s : scores) {
        int i = std::get<1>(s);
        int j = std::get<2>(s);

        if (!used1[i] && !used2[j]) {
            Connection conn;
            conn.part_idx1 = i;
            conn.part_idx2 = j;
            conn.score = std::get<0>(s);
            connections.push_back(conn);

            used1[i] = true;
            used2[j] = true;
        }
    }
}


/**
 * 키포인트들을 사람별로 그룹화
 */
static void group_keypoints(
    const std::vector<std::vector<Keypoint>>& all_keypoints,
    const std::vector<std::vector<Connection>>& all_connections,
    std::vector<HumanPose>& humans)
{
    // 각 키포인트의 사람 ID를 추적
    std::vector<std::vector<int>> keypoint_to_human(NUM_KEYPOINTS);
    for (int p = 0; p < NUM_KEYPOINTS; p++) {
        keypoint_to_human[p].resize(all_keypoints[p].size(), -1);
    }

    int next_human_id = 0;

    // 각 연결에 대해 처리
    for (int link_idx = 0; link_idx < NUM_LINKS; link_idx++) {
        int part1 = SKELETON[link_idx][0];
        int part2 = SKELETON[link_idx][1];

        for (const auto& conn : all_connections[link_idx]) {
            int idx1 = conn.part_idx1;
            int idx2 = conn.part_idx2;

            int human_id1 = keypoint_to_human[part1][idx1];
            int human_id2 = keypoint_to_human[part2][idx2];

            if (human_id1 < 0 && human_id2 < 0) {
                // 둘 다 할당되지 않음 - 새 사람 생성
                keypoint_to_human[part1][idx1] = next_human_id;
                keypoint_to_human[part2][idx2] = next_human_id;
                next_human_id++;
            } else if (human_id1 >= 0 && human_id2 < 0) {
                // 첫 번째만 할당됨
                keypoint_to_human[part2][idx2] = human_id1;
            } else if (human_id1 < 0 && human_id2 >= 0) {
                // 두 번째만 할당됨
                keypoint_to_human[part1][idx1] = human_id2;
            } else if (human_id1 != human_id2) {
                // 병합 필요 (같은 사람의 다른 부분)
                int merge_to = std::min(human_id1, human_id2);
                int merge_from = std::max(human_id1, human_id2);

                for (int p = 0; p < NUM_KEYPOINTS; p++) {
                    for (size_t k = 0; k < keypoint_to_human[p].size(); k++) {
                        if (keypoint_to_human[p][k] == merge_from) {
                            keypoint_to_human[p][k] = merge_to;
                        }
                    }
                }
            }
        }
    }

    // 사람별로 키포인트 수집
    std::map<int, HumanPose> human_map;

    for (int p = 0; p < NUM_KEYPOINTS; p++) {
        for (size_t k = 0; k < all_keypoints[p].size(); k++) {
            int human_id = keypoint_to_human[p][k];
            if (human_id >= 0) {
                if (human_map.find(human_id) == human_map.end()) {
                    HumanPose pose;
                    pose.score = 0;
                    pose.num_parts = 0;
                    for (int i = 0; i < NUM_KEYPOINTS; i++) {
                        pose.keypoints[i].score = -1;
                    }
                    human_map[human_id] = pose;
                }

                const Keypoint& kp = all_keypoints[p][k];
                human_map[human_id].keypoints[p] = kp;
                human_map[human_id].score += kp.score;
                human_map[human_id].num_parts++;
            }
        }
    }

    // 결과 수집
    for (auto& pair : human_map) {
        HumanPose& pose = pair.second;
        if (pose.num_parts >= 4) {  // 최소 4개 키포인트 필요
            pose.score /= pose.num_parts;
            humans.push_back(pose);
        }
    }
}


/**
 * DeepStream 커스텀 포즈 파서
 */
extern "C" bool NvDsInferParsePose(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    // 출력 레이어 찾기 (cmap: heatmap, paf: part affinity field)
    const float* cmap = nullptr;
    const float* paf = nullptr;
    int cmap_c = 0, cmap_h = 0, cmap_w = 0;
    int paf_c = 0, paf_h = 0, paf_w = 0;

    for (const auto& layer : outputLayersInfo) {
        if (strcmp(layer.layerName, "cmap") == 0) {
            cmap = static_cast<const float*>(layer.buffer);
            cmap_c = layer.inferDims.d[0];  // NUM_KEYPOINTS
            cmap_h = layer.inferDims.d[1];
            cmap_w = layer.inferDims.d[2];
        } else if (strcmp(layer.layerName, "paf") == 0) {
            paf = static_cast<const float*>(layer.buffer);
            paf_c = layer.inferDims.d[0];  // 2 * NUM_LINKS
            paf_h = layer.inferDims.d[1];
            paf_w = layer.inferDims.d[2];
        }
    }

    if (!cmap || !paf) {
        // 레이어를 찾지 못함
        return false;
    }

    // 1. 각 키포인트 타입별로 후보 추출 (NMS)
    std::vector<std::vector<Keypoint>> all_keypoints(NUM_KEYPOINTS);

    for (int p = 0; p < NUM_KEYPOINTS && p < cmap_c; p++) {
        const float* heatmap = cmap + p * cmap_h * cmap_w;
        nms_heatmap(heatmap, cmap_h, cmap_w, p, HEATMAP_THRESHOLD, all_keypoints[p]);
    }

    // 2. PAF를 이용하여 연결 점수 계산 및 매칭
    std::vector<std::vector<Connection>> all_connections(NUM_LINKS);

    for (int link = 0; link < NUM_LINKS; link++) {
        int part1 = SKELETON[link][0];
        int part2 = SKELETON[link][1];

        if (part1 >= NUM_KEYPOINTS || part2 >= NUM_KEYPOINTS) continue;
        if (all_keypoints[part1].empty() || all_keypoints[part2].empty()) continue;

        int paf_idx_x = PAF_INDEX[link][0];
        int paf_idx_y = PAF_INDEX[link][1];

        if (paf_idx_x >= paf_c || paf_idx_y >= paf_c) continue;

        const float* paf_x = paf + paf_idx_x * paf_h * paf_w;
        const float* paf_y = paf + paf_idx_y * paf_h * paf_w;

        // 점수 행렬 계산
        std::vector<std::vector<float>> score_matrix(
            all_keypoints[part1].size(),
            std::vector<float>(all_keypoints[part2].size(), 0.0f)
        );

        for (size_t i = 0; i < all_keypoints[part1].size(); i++) {
            for (size_t j = 0; j < all_keypoints[part2].size(); j++) {
                score_matrix[i][j] = compute_paf_score(
                    paf_x, paf_y, paf_h, paf_w,
                    all_keypoints[part1][i],
                    all_keypoints[part2][j]
                );
            }
        }

        // 탐욕적 매칭
        greedy_matching(
            all_keypoints[part1],
            all_keypoints[part2],
            score_matrix,
            PAF_SCORE_THRESHOLD,
            all_connections[link]
        );
    }

    // 3. 키포인트를 사람별로 그룹화
    std::vector<HumanPose> humans;
    group_keypoints(all_keypoints, all_connections, humans);

    // 4. DeepStream 객체 메타데이터 생성
    for (const auto& human : humans) {
        // 바운딩 박스 계산
        float min_x = 1.0f, min_y = 1.0f, max_x = 0.0f, max_y = 0.0f;

        for (int p = 0; p < NUM_KEYPOINTS; p++) {
            if (human.keypoints[p].score >= 0) {
                min_x = std::min(min_x, human.keypoints[p].x);
                min_y = std::min(min_y, human.keypoints[p].y);
                max_x = std::max(max_x, human.keypoints[p].x);
                max_y = std::max(max_y, human.keypoints[p].y);
            }
        }

        // 박스 여백 추가
        float padding = 0.05f;
        min_x = std::max(0.0f, min_x - padding);
        min_y = std::max(0.0f, min_y - padding);
        max_x = std::min(1.0f, max_x + padding);
        max_y = std::min(1.0f, max_y + padding);

        NvDsInferObjectDetectionInfo obj;
        obj.classId = 0;  // person
        obj.detectionConfidence = human.score;
        obj.left = min_x * networkInfo.width;
        obj.top = min_y * networkInfo.height;
        obj.width = (max_x - min_x) * networkInfo.width;
        obj.height = (max_y - min_y) * networkInfo.height;

        objectList.push_back(obj);
    }

    return true;
}


// 모듈 검증 함수
extern "C" bool NvDsInferClassiferParseCustomSoftmax(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute>& attrList,
    std::string& descString)
{
    return false;  // 분류기 파서 미사용
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParsePose);
