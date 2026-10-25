import http.cookiejar
import json
import os
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request


@unittest.skipUnless(shutil.which('openssl'), 'OpenSSL is required to create a temporary HTTPS test certificate')
class LanSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.cert = root / 'localhost-cert.pem'
        cls.key = root / 'localhost-key.pem'
        cls.workspace = root / 'workspace'
        cls.workspace.mkdir()
        subprocess.run([shutil.which('openssl'), 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
            '-keyout', str(cls.key), '-out', str(cls.cert), '-days', '1', '-subj', '/CN=127.0.0.1',
            '-addext', 'subjectAltName=IP:127.0.0.1'], check=True, capture_output=True)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            cls.port = sock.getsockname()[1]
        cls.base = f'https://127.0.0.1:{cls.port}'
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
        env['TRAFFIC_ACCESS_CODE'] = 'test-operator-code-0123456789'
        cls.proc = subprocess.Popen([sys.executable, '-m', 'road_traffic', 'serve', '--host', '0.0.0.0',
            '--public-host', '127.0.0.1', '--tls-cert', str(cls.cert), '--tls-key', str(cls.key),
            '--workspace', str(cls.workspace), '--port', str(cls.port)], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        context = ssl._create_unverified_context()
        cls.cookies = http.cookiejar.CookieJar()
        cls.client = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context),
            urllib.request.HTTPCookieProcessor(cls.cookies))
        for _ in range(120):
            if cls.proc.poll() is not None:
                cls.tmp.cleanup()
                raise RuntimeError('LAN server exited before becoming ready')
            try:
                cls.client.open(cls.base + '/api/session', timeout=.25)
            except urllib.error.HTTPError as exc:
                if exc.code == 401:
                    break
                if exc.code == 403:
                    cls.proc.terminate()
                    cls.tmp.cleanup()
                    raise RuntimeError('LAN server rejected its configured public Host') from exc
            except OSError:
                time.sleep(.05)
        else:
            cls.proc.terminate()
            cls.proc.wait(timeout=5)
            cls.tmp.cleanup()
            raise RuntimeError('LAN HTTPS server did not become ready')

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=10)
        cls.tmp.cleanup()

    def test_access_code_session_protects_api_and_logout_revokes_it(self):
        with self.client.open(self.base + '/', timeout=3) as response:
            self.assertIn(b'Traffic Analysis Sign In', response.read())
        with self.assertRaises(urllib.error.HTTPError) as denied:
            self.client.open(self.base + '/api/session', timeout=3)
        self.assertEqual(denied.exception.code, 401)

        def sign_in(code):
            body = urllib.parse.urlencode({'code': code}).encode()
            request = urllib.request.Request(self.base + '/api/login', data=body,
                headers={'Origin': self.base, 'Content-Type': 'application/x-www-form-urlencoded'})
            return self.client.open(request, timeout=3)

        with self.assertRaises(urllib.error.HTTPError) as wrong_code:
            sign_in('incorrect-operator-code')
        self.assertEqual(wrong_code.exception.code, 401)
        with sign_in('test-operator-code-0123456789') as response:
            self.assertEqual(response.status, 200)
        with self.client.open(self.base + '/api/session', timeout=3) as response:
            session = json.load(response)
        self.assertEqual(session['mode'], 'authenticated-lan')
        with self.client.open(self.base + '/', timeout=3) as response:
            self.assertIn(b'id="lanLogout"', response.read())

        body = json.dumps({'name': 'Unauthorized mutation', 'country': 'Israel',
            'zone': 'Asia/Jerusalem'}).encode()
        request = urllib.request.Request(self.base + '/api/missions', data=body,
            headers={'Origin': self.base, 'Content-Type': 'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as missing_token:
            self.client.open(request, timeout=3)
        self.assertEqual(missing_token.exception.code, 403)

        request = urllib.request.Request(self.base + '/api/logout', data=b'',
            headers={'Origin': self.base, 'X-Session-Token': session['token']})
        with self.client.open(request, timeout=3) as response:
            self.assertEqual(response.status, 200)
        with self.assertRaises(urllib.error.HTTPError) as revoked:
            self.client.open(self.base + '/api/session', timeout=3)
        self.assertEqual(revoked.exception.code, 401)

    def test_invalid_origin_cannot_create_a_session(self):
        body = urllib.parse.urlencode({'code': 'test-operator-code-0123456789'}).encode()
        request = urllib.request.Request(self.base + '/api/login', data=body,
            headers={'Origin': 'https://attacker.invalid',
                'Content-Type': 'application/x-www-form-urlencoded'})
        with self.assertRaises(urllib.error.HTTPError) as rejected:
            self.client.open(request, timeout=3)
        self.assertEqual(rejected.exception.code, 403)


if __name__ == '__main__':
    unittest.main()
