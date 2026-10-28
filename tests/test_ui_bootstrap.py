from html.parser import HTMLParser
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MissionWizardMarkup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_mission_form = False
        self.step_count = 0
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == 'form' and attributes.get('id') == 'missionForm':
            self.in_mission_form = True
        if not self.in_mission_form:
            return
        if tag == 'fieldset' and 'mission-wizard-step' in attributes.get('class', '').split():
            self.step_count += 1
        if attributes.get('id'):
            self.ids.add(attributes['id'])

    def handle_endtag(self, tag):
        if tag == 'form' and self.in_mission_form:
            self.in_mission_form = False


class UiBootstrapTests(unittest.TestCase):
    def test_translation_table_is_initialized_before_overrides(self):
        source = (ROOT / 'web' / 'app.js').read_text(encoding='utf-8')
        declaration = source.index("const en={};document.querySelectorAll('[data-t]')")
        first_override = source.index('Object.assign(en,')
        self.assertLess(declaration, first_override)

    def test_mission_configuration_is_a_three_step_wizard(self):
        parser = MissionWizardMarkup()
        parser.feed((ROOT / 'web' / 'index.html').read_text(encoding='utf-8'))
        self.assertEqual(parser.step_count, 3)
        self.assertTrue({
            'missionWizardStatus', 'missionWizardProgress', 'missionWizardBack',
            'missionWizardNext', 'createMission', 'field2', 'inputType',
            'analysisMode', 'field3', 'location', 'field4', 'sourceDirectory',
            'outputDirectory',
        }.issubset(parser.ids))


if __name__ == '__main__':
    unittest.main()
