# 연결 문제 해결 가이드

## ERR_CONNECTION_REFUSED 오류 해결

### 1. 서버가 실행 중인지 확인

```bash
# Python 프로세스 확인
ps aux | grep python | grep main.py

# 포트 사용 확인
netstat -tulpn | grep 5000
# 또는
lsof -i :5000
```

### 2. 서버 시작 확인

서버를 시작할 때 다음과 같은 메시지가 나타나야 합니다:

```
==================================================
웹 스트리밍 서버 시작 중...
==================================================
호스트: 0.0.0.0
포트: 5000
서버 IP: 192.168.x.x

브라우저에서 다음 주소로 접속하세요:
  🌐 http://192.168.x.x:5000
  🖥️  http://localhost:5000 (같은 머신에서)
==================================================

✅ 서버가 시작되었습니다!
```

**서버가 시작되지 않았다면:**
- `--web` 옵션을 사용했는지 확인
- 오류 메시지를 확인
- 비디오 소스가 올바른지 확인

### 3. 올바른 주소로 접속 확인

**같은 머신에서 접속하는 경우:**
```
http://localhost:5000
또는
http://127.0.0.1:5000
```

**원격에서 접속하는 경우:**
- 서버가 출력한 실제 IP 주소 사용
- 예: `http://192.168.1.100:5000`

### 4. 포트 충돌 해결

포트가 이미 사용 중인 경우:

```bash
# 방법 1: 다른 포트 사용
python3 main.py --web --port 8080

# 방법 2: 기존 프로세스 종료
lsof -ti:5000 | xargs kill -9
# 또는
sudo fuser -k 5000/tcp
```

### 5. 방화벽 확인

```bash
# UFW 방화벽 상태 확인
sudo ufw status

# 포트 열기 (필요한 경우)
sudo ufw allow 5000/tcp

# iptables 확인
sudo iptables -L -n | grep 5000
```

### 6. 서버 연결 확인 스크립트 사용

```bash
# requests 라이브러리 설치 (없는 경우)
pip install requests

# 서버 확인
python3 check_server.py http://localhost:5000
# 또는
python3 check_server.py 5000
```

### 7. 일반적인 문제와 해결책

#### 문제: "Address already in use"
**해결:**
```bash
# 다른 포트 사용
python3 main.py --web --port 8080
```

#### 문제: "Permission denied" (포트 1024 미만)
**해결:**
```bash
# 1024 이상의 포트 사용
python3 main.py --web --port 8080
```

#### 문제: 비디오 소스 오류로 서버가 시작되지 않음
**해결:**
```bash
# 카메라 확인
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL'); cap.release()"

# 비디오 파일 경로 확인
ls -la /path/to/video.mp4
```

#### 문제: SSH 터널링 환경
**해결:**
```bash
# SSH 포트 포워딩 사용
ssh -L 5000:localhost:5000 user@server

# 그 다음 로컬 브라우저에서
http://localhost:5000
```

### 8. 디버그 모드로 실행

더 자세한 정보를 보려면:

```bash
python3 main.py --web --use-yolo --source 0 --debug
```

### 9. 네트워크 진단

```bash
# 서버 IP 확인
hostname -I
# 또는
ip addr show

# 로컬 루프백 확인
ping localhost

# 포트 연결 테스트
telnet localhost 5000
# 또는
nc -zv localhost 5000
```

### 10. 최소 설정으로 테스트

문제를 격리하기 위해 최소 설정으로 테스트:

```bash
# 가장 간단한 설정
python3 main.py --web --source 0 --port 8080

# MediaPipe 사용 (YOLO 없이)
python3 main.py --web --source 0

# 비디오 파일 사용
python3 main.py --web --source test_video.mp4
```

## 추가 도움말

문제가 계속되면 다음 정보를 확인하세요:

1. Python 버전: `python3 --version`
2. 필요한 패키지 설치: `pip install -r requirements.txt`
3. 시스템 로그: `journalctl -xe` (systemd 사용 시)
4. 브라우저 콘솔 오류 (F12 개발자 도구)
