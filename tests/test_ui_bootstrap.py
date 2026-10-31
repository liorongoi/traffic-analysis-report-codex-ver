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


class ReviewLayoutMarkup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.div_classes = []
        self.nested_sectionheads = 0

    def handle_starttag(self, tag, attrs):
        if tag != 'div':
            return
        classes = set(dict(attrs).get('class', '').split())
        if 'sectionhead' in classes and any('sectionhead' in parent for parent in self.div_classes):
            self.nested_sectionheads += 1
        self.div_classes.append(classes)

    def handle_endtag(self, tag):
        if tag == 'div' and self.div_classes:
            self.div_classes.pop()


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

    def test_review_header_does_not_wrap_the_entire_workflow_as_a_flex_row(self):
        parser = ReviewLayoutMarkup()
        parser.feed((ROOT / 'web' / 'index.html').read_text(encoding='utf-8'))
        self.assertEqual(parser.nested_sectionheads, 0)

    def test_video_mode_hides_only_the_image_workflow_container(self):
        markup = (ROOT / 'web' / 'index.html').read_text(encoding='utf-8')
        source = (ROOT / 'web' / 'app.js').read_text(encoding='utf-8')
        styles = (ROOT / 'web' / 'style.css').read_text(encoding='utf-8')
        self.assertIn('<div id="imageWorkflow">', markup)
        self.assertIn("$('videoWorkflow').hidden=!video", source)
        self.assertIn("$('imageWorkflow').hidden=video||!selected", source)
        self.assertIn("$('emptyWorkflow').hidden=!!selected", source)
        self.assertIn('.monitoring-panel[hidden]{display:none}', styles)
        self.assertIn('.empty-workflow[hidden]{display:none}', styles)

    def test_vehicle_model_selector_is_available_outside_the_image_only_panel(self):
        markup = (ROOT / 'web' / 'index.html').read_text(encoding='utf-8')
        source = (ROOT / 'web' / 'app.js').read_text(encoding='utf-8')
        detection_panel = markup.index('id="vehicleDetectionPanel"')
        image_panel_end = markup.index('</details>', detection_panel)
        model_panel = markup.index('id="vehicleModelPanel"')
        search_panel = markup.index('id="objectSearchPanel"')
        self.assertLess(image_panel_end, model_panel)
        self.assertLess(model_panel, search_panel)
        self.assertIn("$('vehicleModelPanel').hidden=!selected||selected.analysis_mode==='license_plate'", source)
        self.assertIn('Model for new still-image and video runs', source)
        self.assertIn('מודל להרצות תמונה ווידאו חדשות', source)

    def test_operator_help_is_translated_to_hebrew(self):
        source = (ROOT / 'web' / 'app.js').read_text(encoding='utf-8')
        for value in (
                "helpTitle:'עזרה למפעיל'", "helpImportTitle:'ייבוא וזמני צילום'",
                "helpScheduleTitle:'חלונות זמן'", "helpViewsTitle:'תצוגות ואזורים'",
                "helpReviewTitle:'בדיקה וייצוא'", "helpScopeTitle:'יכולות נוכחיות'",
                "keyboardHelp:'השתמשו בחצים"):
            self.assertIn(value, source)

    def test_empty_workspace_message_is_bilingual(self):
        source = (ROOT / 'web' / 'app.js').read_text(encoding='utf-8')
        self.assertIn("emptyWorkflowTitle:'Create or select a mission'", source)
        self.assertIn("emptyWorkflowTitle:'צרו או בחרו משימה'", source)

    def test_sam_button_requires_an_image_assigned_to_the_selected_view(self):
        source = (ROOT / 'web' / 'app.js').read_text(encoding='utf-8')
        self.assertIn("canSegment=!!mission&&!!row?.view_id&&row.view_id===$('imageView').value", source)
        self.assertIn("$('suggestRoadMasks').disabled=!canSegment", source)


if __name__ == '__main__':
    unittest.main()
