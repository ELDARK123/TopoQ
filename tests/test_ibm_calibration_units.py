import json
import tempfile
import unittest
from pathlib import Path

import generate_dataset


class IbmCalibrationUnitTests(unittest.TestCase):
    def test_explicit_unit_conversion(self):
        self.assertAlmostEqual(
            generate_dataset.convert_calibration_value(1.0, 's', 'us'), 1e6
        )
        self.assertAlmostEqual(
            generate_dataset.convert_calibration_value(5.113287e9, 'Hz', 'GHz'),
            5.113287,
        )
        self.assertAlmostEqual(
            generate_dataset.convert_calibration_value(5351.111, 'ns', 'ns'),
            5351.111,
        )

    def test_guadalupe_reference_values(self):
        snapshot_dir = generate_dataset.DEFAULT_IBM_SNAPSHOT_DIR
        if not snapshot_dir.is_dir():
            self.skipTest('IBM calibration snapshots have not been localized')

        with tempfile.TemporaryDirectory() as temporary_dir:
            chips = generate_dataset.extract_ibm_chips(
                temporary_dir, str(snapshot_dir)
            )
            qubit_zero = chips['ibm_guadalupe']['qubit_properties'][0]
            self.assertAlmostEqual(qubit_zero['t1_us'], 44.8664962391536)
            self.assertAlmostEqual(qubit_zero['t2_us'], 77.06572105507512)
            self.assertAlmostEqual(qubit_zero['freq_ghz'], 5.113287233682569)
            self.assertAlmostEqual(qubit_zero['readout_error'], 0.0136)
            self.assertAlmostEqual(
                qubit_zero['readout_length_ns'], 5351.11111111111
            )

            generated_path = Path(temporary_dir) / 'ibm_chips' / 'ibm_guadalupe.json'
            with generated_path.open(encoding='utf-8') as stream:
                generated = json.load(stream)
            self.assertEqual(
                generated['calibration_metadata']['standardized_units']['t1_us'],
                'us',
            )

    def test_all_snapshots_pass_physical_range_checks(self):
        snapshot_dir = generate_dataset.DEFAULT_IBM_SNAPSHOT_DIR
        if not snapshot_dir.is_dir():
            self.skipTest('IBM calibration snapshots have not been localized')

        with tempfile.TemporaryDirectory() as temporary_dir:
            chips = generate_dataset.extract_ibm_chips(
                temporary_dir, str(snapshot_dir)
            )
            self.assertEqual(set(chips), set(generate_dataset.IBM_CHIP_SNAPSHOTS))
            for chip in chips.values():
                self.assertGreater(chip['stats']['avg_t1_us'], 0)
                self.assertGreater(chip['stats']['avg_t2_us'], 0)
                self.assertGreater(chip['stats']['avg_cx_duration_ns'], 0)


if __name__ == '__main__':
    unittest.main()
