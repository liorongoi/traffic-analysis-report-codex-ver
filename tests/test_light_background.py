from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from road_traffic.light_background import build_view_backgrounds,load_background


class LightBackgroundTests(unittest.TestCase):
    def test_backgrounds_are_isolated_per_view_and_bounded(self):
        root=Path(tempfile.mkdtemp())
        try:
            rng=np.random.default_rng(7)
            bases={'view-a':rng.integers(0,256,(24,32),dtype=np.uint8),
                   'view-b':rng.integers(0,256,(24,32),dtype=np.uint8)}
            rows=[]
            pixels={}
            for key,count in (('view-a',40),('view-b',5)):
                for index in range(count):
                    identifier=f'{key}-{index:02d}'
                    gray=np.clip(bases[key].astype(np.int16)+(index%3),0,255).astype(np.uint8)
                    rgb=np.repeat(gray[:,:,None],3,axis=2)
                    rows.append({'id':identifier,'sha256':identifier,'view':key})
                    pixels[identifier]=rgb
            artifacts=build_view_backgrounds(rows,{'view-a-00'},lambda row:row['view'],
                lambda row:pixels[row['id']],root,lambda:False,
                cv2_module=cv2,light_mask_fn=lambda gray:(gray>220).astype(np.uint8),
                illumination_fn=lambda rgb:{'low_light':True})
            self.assertEqual(set(artifacts),{'view-a','view-b'})
            self.assertEqual(len(artifacts['view-a']['references']),32)
            self.assertNotIn('view-a-00',[item['image_id'] for item in artifacts['view-a']['references']])
            self.assertTrue(all(item['image_id'].startswith('view-a-')
                for item in artifacts['view-a']['references']))
            self.assertEqual(len(artifacts['view-b']['references']),5)
            saved_a=load_background(artifacts['view-a'])
            saved_b=load_background(artifacts['view-b'])
            self.assertEqual(saved_a['counts'].shape,(24,32))
            self.assertEqual(saved_b['counts'].shape,(24,32))
            self.assertEqual(set(saved_a['reference_ids'].tolist()),
                {item['image_id'] for item in artifacts['view-a']['references']})
            self.assertNotEqual(artifacts['view-a']['sha256'],artifacts['view-b']['sha256'])
        finally:
            import shutil
            shutil.rmtree(root)


if __name__=='__main__':
    unittest.main()
