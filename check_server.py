#!/usr/bin/env python3
"""
서버 연결 확인 스크립트
웹 서버가 정상적으로 실행 중인지 확인
"""

import sys
import socket
import requests
from urllib.parse import urlparse

def check_port(host, port):
    """포트가 열려있는지 확인"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"포트 확인 오류: {e}")
        return False

def check_server(url):
    """서버가 응답하는지 확인"""
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionRefusedError:
        return False
    except requests.exceptions.Timeout:
        return False
    except Exception as e:
        print(f"서버 확인 오류: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("사용법: python3 check_server.py <URL 또는 포트>")
        print("예시:")
        print("  python3 check_server.py http://localhost:5000")
        print("  python3 check_server.py 5000")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    # URL 형식인지 포트 번호인지 확인
    if arg.startswith('http://') or arg.startswith('https://'):
        url = arg
        parsed = urlparse(url)
        host = parsed.hostname or 'localhost'
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    else:
        try:
            port = int(arg)
            host = 'localhost'
            url = f"http://{host}:{port}"
        except ValueError:
            print(f"오류: 잘못된 입력 '{arg}'")
            print("URL 또는 포트 번호를 입력하세요.")
            sys.exit(1)
    
    print(f"\n{'='*50}")
    print("서버 연결 확인")
    print(f"{'='*50}")
    print(f"호스트: {host}")
    print(f"포트: {port}")
    print(f"URL: {url}")
    print(f"{'='*50}\n")
    
    # 포트 확인
    print("1. 포트 연결 확인 중...")
    if check_port(host, port):
        print(f"   ✅ 포트 {port}가 열려있습니다.")
    else:
        print(f"   ❌ 포트 {port}에 연결할 수 없습니다.")
        print(f"\n   가능한 원인:")
        print(f"   - 서버가 실행되지 않았습니다")
        print(f"   - 방화벽이 포트를 막고 있습니다")
        print(f"   - 잘못된 호스트 주소입니다")
        print(f"\n   확인 방법:")
        print(f"   - 서버가 실행 중인지 확인: ps aux | grep python")
        print(f"   - 포트 사용 확인: netstat -tulpn | grep {port}")
        print(f"   - 방화벽 확인: sudo ufw status")
        sys.exit(1)
    
    # 서버 응답 확인
    print("\n2. 서버 응답 확인 중...")
    if check_server(url):
        print(f"   ✅ 서버가 정상적으로 응답합니다!")
        print(f"\n   브라우저에서 다음 주소로 접속하세요:")
        print(f"   {url}")
    else:
        print(f"   ⚠️  서버가 응답하지 않습니다.")
        print(f"   포트는 열려있지만 서버가 요청을 처리하지 못하고 있습니다.")
        print(f"   서버 로그를 확인하세요.")
        sys.exit(1)
    
    print(f"\n{'='*50}")
    print("✅ 모든 확인 완료!")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
