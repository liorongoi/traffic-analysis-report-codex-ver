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

    def test_plate_help_describes_chip_approval_without_text_recognition(self):
        source = (ROOT / 'web' / 'app.js').read_text(encoding='utf-8')
        self.assertIn('Plate-region detection saves image chips for operator approval or rejection; it does not read plate text.', source)
        self.assertNotIn('OCR', source)

    def test_plate_crop_country_is_presented_as_mission_context_not_inference(self):
        source = (ROOT / 'web' / 'app.js').read_text(encoding='utf-8')
        self.assertIn('item.mission_country_context', source)
        self.assertIn('operator configured, not inferred', source)
        self.assertIn('הוגדר בידי המפעיל, לא זוהה אוטומטית', source)

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

    def test_mission_backup_controls_are_bilingual_and_non_overwriting(self):
        source = (ROOT / 'web' / 'app.js').read_text(encoding='utf-8')
        for value in ('missionBackupPanel', 'missionBackupCreate', 'missionBackupVerify',
                'missionRestore', 'missionBackupSaved', 'missionBackupValid'):
            self.assertIn(value, source)
        self.assertIn('Back up or restore a mission', source)
        self.assertIn('גיבוי או שחזור משימה', source)
        self.assertIn("'/api/missions/backup'", source)
        self.assertIn("'/api/missions/restore'", source)

    def test_sam_button_requires_an_image_assigned_to_the_selected_view(self):
        source = (ROOT / 'web' / 'app.js').read_text(encoding='utf-8')
        self.assertIn("canSegment=!!mission&&!!row?.view_id&&row.view_id===$('imageView').value", source)
        self.assertIn("$('suggestRoadMasks').disabled=!canSegment", source)


if __name__ == '__main__':
    unittest.main()
