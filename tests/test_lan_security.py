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
        cls.proc = subprocess.Popen([sys.executable, '-m', 'road_traffic', 'serve', '--host', '0.0.0.0',
            '--public-host', '127.0.0.1', '--tls-cert', str(cls.cert), '--tls-key', str(cls.key),
            '--workspace', str(cls.workspace), '--port', str(cls.port)], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        context = ssl._create_unverified_context()
        cls.client = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
        for _ in range(120):
            if cls.proc.poll() is not None:
                cls.tmp.cleanup()
                raise RuntimeError('LAN server exited before becoming ready')
            try:
                with cls.client.open(cls.base + '/api/session', timeout=.25) as response:
                    if response.status == 200:
                        break
            except urllib.error.HTTPError as exc:
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

    def test_lan_works_without_sign_in_and_writes_keep_token_protection(self):
        with self.client.open(self.base + '/', timeout=3) as response:
            page = response.read().decode('utf-8')
            self.assertIn('RoadLoom', page)
            self.assertNotIn('lanLogout', page)
            self.assertNotIn('login', page.lower())
            self.assertNotIn('password', page.lower())
        with self.client.open(self.base + '/api/session', timeout=3) as response:
            session = json.load(response)
            self.assertEqual(session['mode'], 'lan')
            self.assertNotIn('Set-Cookie', response.headers)
        body = json.dumps({'name': 'Authorized mission', 'country': 'Israel',
            'zone': 'Asia/Jerusalem'}).encode()
        request = urllib.request.Request(self.base + '/api/missions', data=body,
            headers={'Origin': self.base, 'X-Session-Token': session['token'],
                'Content-Type': 'application/json'})
        with self.client.open(request, timeout=3) as response:
            self.assertEqual(response.status, 200)

    def test_unknown_host_and_mutations_without_token_are_rejected(self):
        request = urllib.request.Request(self.base + '/', headers={'Host': 'attacker.invalid'})
        with self.assertRaises(urllib.error.HTTPError) as rejected_host:
            self.client.open(request, timeout=3)
        self.assertEqual(rejected_host.exception.code, 403)
        body = json.dumps({'name': 'Unauthorized mission', 'country': 'Israel',
            'zone': 'Asia/Jerusalem'}).encode()
        request = urllib.request.Request(self.base + '/api/missions', data=body,
            headers={'Origin': self.base, 'Content-Type': 'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as rejected_write:
            self.client.open(request, timeout=3)
        self.assertEqual(rejected_write.exception.code, 403)

    def test_invalid_origin_and_sign_in_routes_are_rejected(self):
        with self.client.open(self.base + '/api/session', timeout=3) as response:
            session = json.load(response)
        body = json.dumps({'name': 'Rejected mission', 'country': 'Israel',
            'zone': 'Asia/Jerusalem'}).encode()
        request = urllib.request.Request(self.base + '/api/missions', data=body,
            headers={'Origin': 'https://attacker.invalid', 'X-Session-Token': session['token'],
                'Content-Type': 'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as rejected_origin:
            self.client.open(request, timeout=3)
        self.assertEqual(rejected_origin.exception.code, 403)
        for path in ('/login', '/login.js', '/api/login', '/api/logout'):
            with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as missing:
                self.client.open(self.base + path, timeout=3)
            self.assertEqual(missing.exception.code, 404)


if __name__ == '__main__':
    unittest.main()
