import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from download_medhop_hf import parquet_urls, mirror_url, save_url, convert_parquet_to_original, CONFIG

class TestHuggingFaceAlternative(unittest.TestCase):
    def test_source_catalog_required(self):
        catalog = {CONFIG: {
            'train':['https://huggingface.co/api/datasets/bigbio/medhop/parquet/medhop_source/train/0.parquet'],
            'validation':['https://huggingface.co/api/datasets/bigbio/medhop/parquet/medhop_source/validation/0.parquet']}}
        self.assertEqual(len(parquet_urls(catalog)['train']),1)
        with self.assertRaisesRegex(ValueError,'source'):
            parquet_urls({'medhop_bigbio_qa':catalog[CONFIG]})
        with self.assertRaisesRegex(ValueError,'validation'):
            parquet_urls({CONFIG:{'train':catalog[CONFIG]['train']}})

    def test_mirror_only_changes_huggingface_hosts(self):
        src='https://huggingface.co/datasets/bigbio/medhop/resolve/main/train.parquet?download=1'
        self.assertEqual(mirror_url(src,'https://hf-mirror.com'),'https://hf-mirror.com/datasets/bigbio/medhop/resolve/main/train.parquet?download=1')
        self.assertEqual(mirror_url('https://unrelated.example/data','https://hf-mirror.com'),'https://unrelated.example/data')

    def test_failed_download_never_leaves_complete_destination(self):
        with tempfile.TemporaryDirectory() as t:
            dest=Path(t)/'train.parquet'
            with patch('download_medhop_hf.urllib.request.urlopen',side_effect=OSError('network down')):
                with self.assertRaises(OSError):
                    save_url('https://huggingface.co/foo',dest,'https://huggingface.co')
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_suffix('.parquet.part').exists())

    def test_parquet_transform_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            dest=Path(t)
            (dest/'medhop_original').mkdir()
            (dest/'medhop_original'/'train.json').write_text('[]')
            # Missing pyarrow is allowed, this test only checks pre-existing data invariant.
            # This is checked by the higher-level download_and_convert() guard.
            from download_medhop_hf import download_and_convert
            with self.assertRaises(FileExistsError):
                download_and_convert(dest,['https://huggingface.co'])

if __name__=='__main__':unittest.main()
