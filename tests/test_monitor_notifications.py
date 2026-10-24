import tempfile
import unittest
from pathlib import Path

from road_traffic.monitor import SourceMonitor


class FakeLibrary:
    def __init__(self, mission):
        self.mission = mission
        self.enabled = True

    def missions(self):
        return [self.mission]

    def monitoring_enabled(self, mission_id):
        return self.enabled


class FakeObserver:
    def __init__(self):
        self.handler = None
        self.path = None
        self.recursive = None
        self.started = False
        self.stopped = False

    def schedule(self, handler, path, recursive):
        self.handler, self.path, self.recursive = handler, path, recursive

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def join(self, timeout=None):
        return None


class MonitorNotificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'source'
        self.root.mkdir()
        self.mission = {'id': 'mission-1', 'processing_mode': 'continuous', 'input_type': 'images',
                        'source_root': str(self.root), 'include_subdirectories': True}
        self.library = FakeLibrary(self.mission)
        self.created = []

        def factory():
            observer = FakeObserver()
            self.created.append(observer)
            return observer

        self.monitor = SourceMonitor(self.library, observer_factory=factory)

    def tearDown(self):
        self.monitor.stop()
        self.temp.cleanup()

    def test_file_notifications_wake_scanner_and_reconciliation_watch_is_recursive(self):
        self.monitor._refresh_observers(self.library.missions())
        observer = self.created[0]
        self.assertTrue(observer.started)
        self.assertEqual(observer.path, str(self.root.resolve()))
        self.assertTrue(observer.recursive)
        self.monitor.wake_event.clear()
        observer.handler.on_any_event(object())
        self.assertTrue(self.monitor.wake_event.is_set())
        self.assertEqual(self.monitor.notification_status['mission-1']['status'], 'active')

    def test_watch_is_rebuilt_when_recursion_changes_and_removed_on_pause(self):
        self.monitor._refresh_observers(self.library.missions())
        first = self.created[0]
        self.mission['include_subdirectories'] = False
        self.monitor._refresh_observers(self.library.missions())
        self.assertTrue(first.stopped)
        second = self.created[1]
        self.assertFalse(second.recursive)
        self.library.enabled = False
        self.monitor._refresh_observers(self.library.missions())
        self.assertTrue(second.stopped)
        self.assertEqual(self.monitor.notification_status['mission-1']['status'], 'paused')


    def test_unconfigured_source_does_not_watch_the_current_directory(self):
        self.mission['source_root'] = ''
        self.monitor._refresh_observers(self.library.missions())
        self.assertEqual(self.created, [])
        self.assertEqual(self.monitor.notification_status['mission-1']['status'], 'unconfigured')

    def test_observer_failure_keeps_periodic_reconciliation_available(self):
        def fail():
            raise OSError('watch handle limit')
        monitor = SourceMonitor(self.library, observer_factory=fail)
        monitor._refresh_observers(self.library.missions())
        self.assertEqual(monitor.notification_status['mission-1']['status'], 'polling_fallback')
        self.assertEqual(monitor.notification_status['mission-1']['error'], 'watch handle limit')
