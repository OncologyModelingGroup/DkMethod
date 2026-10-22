"""Fast orchestration tests; no BraTS data or PDE simulation required."""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np


RUNNER = Path(__file__).resolve().parents[1] / "Pipeline" / "run.py"
spec = importlib.util.spec_from_file_location("dk_pipeline_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.calls = []
        subject = self.output / "P1"
        subject.mkdir()
        (subject / "BraTS17_2013_001_1_ManuallyCorrected.nii.gz").touch()
        (subject / "BraTS17_2013_001_1_flair.nii.gz").touch()

        patient = types.ModuleType("ProcessPatientData")
        patient.GetLargestCom = object()
        patient.FindSegPath = lambda files: next((name for name in files if "_ManuallyCorrected." in name or "_GlistrBoost." in name), None)
        patient.FindMaskPath = lambda files: next((name for name in files if "_flair." in name), None)
        def paths(root, destination):
            self.calls.append("paths")
            pd.to_pickle({"P1": {"root": root}}, destination)
        def patients(data, destination, com):
            self.calls.append("patients")
            pd.to_pickle({"P1": "counts"}, destination)
        patient.GenBratsPaths = paths
        patient.ProcessPts = patients

        model = types.ModuleType("Model3Dand1D")
        model.ModelParams = lambda d, **kwargs: (d, kwargs)
        def grid(params, destination, saveFreq):
            self.calls.append("grid")
            self.assertEqual(len(params), 2)
            pd.to_pickle([types.SimpleNamespace(params=types.SimpleNamespace(d=d, **kwargs), saveFreq=saveFreq) for d, kwargs in params], destination)
        model.GenGrid = grid

        fitting = types.ModuleType("FitDkForEachPatient")
        def fits(data, grid, destination):
            self.calls.append("fits")
            self.assertEqual(data, {"P1": "counts"})
            self.assertEqual(len(grid), 2)
            pd.to_pickle(["fit"], destination)
        def table(fits, outPath):
            self.calls.append("table")
            self.assertEqual(fits, ["fit"])
            pd.DataFrame({"D/k": [1.0]}).to_csv(outPath, index=False)
        fitting.GenPatientDks = fits
        fitting.GenPatientDkResDf = table

        modules = {"ProcessPatientData": patient, "Model3Dand1D": model, "FitDkForEachPatient": fitting}
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def args(self, *extra):
        return runner.parse_args(["--brats", str(self.output), "--output", str(self.output), "--n-d", "2", *extra])

    def test_full_patient_ids_are_preserved(self):
        scans = self.output / "scans"
        for subject in ("BraTS17_2013_001_1", "BraTS17_2013_002_1"):
            folder = scans / subject
            folder.mkdir(parents=True)
            (folder / f"{subject}_ManuallyCorrected.nii.gz").touch()
            (folder / f"{subject}_flair.nii.gz").touch()
        runner.run(self.args("--brats", str(scans), "--to-stage", "paths"))
        paths = pd.read_pickle(self.output / "paths.p")
        self.assertEqual(set(paths), {"BraTS17_2013_001_1", "BraTS17_2013_002_1"})
        self.assertEqual({record["pt"] for record in paths.values()}, set(paths))
        self.assertEqual(self.calls, [])

    def test_duplicate_subject_directory_names_are_rejected(self):
        scans = self.output / "scans"
        for group in ("HGG", "LGG"):
            folder = scans / group / "BraTS17_2013_001_1"
            folder.mkdir(parents=True)
            (folder / "BraTS17_2013_001_1_ManuallyCorrected.nii.gz").touch()
            (folder / "BraTS17_2013_001_1_flair.nii.gz").touch()
        with self.assertRaisesRegex(ValueError, "Duplicate BraTS subject directory name"):
            runner.run(self.args("--brats", str(scans), "--to-stage", "paths"))
        self.assertFalse((self.output / "paths.p").exists())

    def test_missing_mask_is_rejected(self):
        scans = self.output / "scans"
        folder = scans / "BraTS17_2013_001_1"
        folder.mkdir(parents=True)
        (folder / "BraTS17_2013_001_1_ManuallyCorrected.nii.gz").touch()
        with self.assertRaisesRegex(ValueError, "No matching MRI mask"):
            runner.run(self.args("--brats", str(scans), "--to-stage", "paths"))

    def test_complete_pipeline_and_checkpoint_reuse(self):
        runner.run(self.args())
        self.assertEqual(self.calls, ["patients", "grid", "fits", "table"])
        self.assertTrue((self.output / "dkDf.csv").exists())
        runner.run(self.args())
        self.assertEqual(self.calls, ["patients", "grid", "fits", "table"])

    def test_changed_grid_parameters_rejected_on_resume(self):
        runner.run(self.args())
        with self.assertRaisesRegex(ValueError, "different model parameters"):
            runner.run(self.args("--k", "0.0002"))

    def test_missing_upstream_checkpoint(self):
        with self.assertRaisesRegex(FileNotFoundError, "requires"):
            runner.run(self.args("--from-stage", "table"))

    def test_invalid_stage_order(self):
        with self.assertRaisesRegex(ValueError, "precede"):
            runner.run(self.args("--from-stage", "table", "--to-stage", "paths"))

    def test_resume_from_table(self):
        pd.to_pickle(["fit"], self.output / "dks.p")
        runner.run(self.args("--from-stage", "table"))
        self.assertEqual(self.calls, ["table"])

    def test_empty_patient_discovery_fails(self):
        empty = self.output / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "No segmented BraTS patients"):
            runner.run(self.args("--brats", str(empty), "--to-stage", "paths"))
        self.assertFalse((self.output / "paths.p").exists())

    def test_missing_brats_directory_fails(self):
        with self.assertRaises(NotADirectoryError):
            runner.run(self.args("--brats", str(self.output / "not-present"), "--to-stage", "paths"))

    def test_empty_processed_patients_fails(self):
        import ProcessPatientData as patient
        patient.ProcessPts = lambda data, destination, com: pd.to_pickle({}, destination)
        with self.assertRaisesRegex(ValueError, "no usable patients"):
            runner.run(self.args("--to-stage", "patients"))
        self.assertFalse((self.output / "pts.p").exists())

    def test_empty_model_grid_fails(self):
        import Model3Dand1D as model
        model.GenGrid = lambda params, destination, saveFreq: pd.to_pickle([], destination)
        with self.assertRaisesRegex(ValueError, "no simulations"):
            runner.run(self.args("--from-stage", "grid", "--to-stage", "grid"))
        self.assertFalse((self.output / "grid.p").exists())

    def test_empty_fits_fails(self):
        import FitDkForEachPatient as fitting
        fitting.GenPatientDks = lambda data, grid, destination: pd.to_pickle({}, destination)
        pd.to_pickle({"P1": "counts"}, self.output / "pts.p")
        pd.to_pickle(["model"], self.output / "grid.p")
        with self.assertRaisesRegex(ValueError, "no estimates"):
            runner.run(self.args("--from-stage", "fits", "--to-stage", "fits"))
        self.assertFalse((self.output / "dks.p").exists())

    def test_empty_results_table_rejected(self):
        import FitDkForEachPatient as fitting
        fitting.GenPatientDkResDf = lambda fits, outPath: pd.DataFrame({"D/k": []}).to_csv(outPath, index=False)
        pd.to_pickle(["fit"], self.output / "dks.p")
        with self.assertRaisesRegex(ValueError, "results table is empty"):
            runner.run(self.args("--from-stage", "table"))
        self.assertFalse((self.output / "dkDf.csv").exists())

    def test_fractional_step_less_than_one_rejected(self):
        with self.assertRaises(SystemExit):
            self.args("--steps", "0.5")

    def test_snapshot_spacing(self):
        with self.assertRaises(SystemExit):
            self.args("--steps", "100", "--save-freq", "100")

    def test_invalid_diffusion_grid(self):
        for options in (("--n-d", "1"), ("--d-min-log", "-2", "--d-max-log", "-5"), ("--dx", "nan")):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                self.args(*options)

    def test_unstable_radial_diffusion_rejected(self):
        with self.assertRaises(SystemExit):
            self.args("--d-max-log", "0")
        with self.assertRaises(SystemExit):
            self.args("--d-max-log", "1000")

    def test_invalid_parameter(self):
        with self.assertRaises(SystemExit):
            self.args("--k", "-1")


if __name__ == "__main__":
    unittest.main()
