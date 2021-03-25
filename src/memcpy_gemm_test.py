# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the memcpy_gemm binary."""

import logging
import os
import re
import subprocess
import unittest

MEMCPY_GEMM_OUTPUT_RE = re.compile(
    r'^(?P<timestamp>[^\t]+)\t(?P<a0>\d+\.\d+)\t(?P<a1>\d+\.\d+)$',
    re.MULTILINE)

# We do not assume anything about performance, just that the output is sensible
# (flow > 0);
MIN_FLOW_RATE = 0.0


class MemcpyGemmTest(unittest.TestCase):

  def setUp(self):  # pylint: disable=g-missing-super-call
    self.duration_s = 3
    self.timeout_s = self.duration_s + 15
    exec_path = 'memcpy_gemm'
    self.memcpy_gemm_args = [
        os.path.abspath(exec_path),
        '--duration={}s'.format(self.duration_s),
        # Tests running on forge can only use 1 GPU.
        '--gpus=0',
        '--flows=c0-g0-a0 g0-c0-a1',
    ]
    self.sm_copy_memcpy_args = [
        os.path.abspath(exec_path),
        '--duration={}s'.format(self.duration_s),
        # Tests running on forge can only use 1 GPU.
        '--gpus=0',
        '--flows=g0-g0-a0 g0-g0-a1',
        '--use_cudacomputecopy',
    ]

  def ValidateOutput(self, output):
    nr_matches = 0
    output_text = output.decode('utf-8')
    logging.info('memcpy_gemm output:\n%s', output_text)
    for m in MEMCPY_GEMM_OUTPUT_RE.finditer(output_text):
      self.assertGreaterEqual(
          float(m['a0']), MIN_FLOW_RATE,
          'At {timestamp}, flow a0 is slower than expected'.format(
              **m.groupdict()))
      nr_matches += 1
    self.assertGreater(nr_matches, 0)

  def testGEMMPrecision(self):
    options = [
        '--fp_precision=half', '--fp_precision=single', '--fp_precision=double'
    ]
    for opt in options:
      args = self.memcpy_gemm_args + ['--gemm', '--N=1024'] + [opt]
      result = subprocess.run(
          args, timeout=self.timeout_s, stdout=subprocess.PIPE)
      self.assertEqual(result.returncode, 0)
      self.ValidateOutput(result.stdout)

  def testMatrixDimensions(self):
    gemm_dimensions = ['--gemm', '--N=1024', '--M=1024', '--K=1024']
    args = self.memcpy_gemm_args + gemm_dimensions
    result = subprocess.run(
        args, timeout=self.timeout_s, stdout=subprocess.PIPE)
    self.assertEqual(result.returncode, 0)
    self.ValidateOutput(result.stdout)

  def testMixedHalfFloat(self):
    args = self.memcpy_gemm_args + [
        '--gemm', '--input_precision=half', '--output_precision=single',
        '--compute_precision=single', '--N=1024', '--M=1024', '--K=1024',
        '--gemm_autotune=1'
    ]
    result = subprocess.run(
        args, timeout=self.timeout_s, stdout=subprocess.PIPE)
    self.assertEqual(result.returncode, 0)
    self.ValidateOutput(result.stdout)

  def testMixedIntInt(self):
    args = self.memcpy_gemm_args + [
        '--gemm', '--input_precision=int8', '--output_precision=int32',
        '--compute_precision=int32', '--N=1024', '--M=1024', '--K=1024'
    ]
    result = subprocess.run(
        args, timeout=self.timeout_s, stdout=subprocess.PIPE)
    self.assertEqual(result.returncode, 0)
    self.ValidateOutput(result.stdout)

  def testMixedIntFloat(self):
    args = self.memcpy_gemm_args + [
        '--gemm', '--input_precision=int8', '--output_precision=single',
        '--compute_precision=single', '--N=1024', '--M=1024', '--K=1024'
    ]
    result = subprocess.run(
        args, timeout=self.timeout_s, stdout=subprocess.PIPE)
    self.assertEqual(result.returncode, 0)
    self.ValidateOutput(result.stdout)

  def testGemmAlgorithms(self):
    args = self.memcpy_gemm_args + [
        '--gemm', '--N=2048', '--algorithm=gemm_algo_3'
    ]
    result = subprocess.run(
        args, timeout=self.timeout_s, stdout=subprocess.PIPE)
    self.assertEqual(result.returncode, 0)
    self.ValidateOutput(result.stdout)

  def testGEMMOptions(self):
    options = [
        '--trigger_period=1',
        '--sync_flows=true',
        '--nv_gaussian=true',
    ]
    for opt in options:
      args = self.memcpy_gemm_args + ['--gemm', '--N=1024'] + [opt]
      result = subprocess.run(
          args, timeout=self.timeout_s, stdout=subprocess.PIPE)
      self.assertEqual(result.returncode, 0)
      self.ValidateOutput(result.stdout)

  def testMemcpyOptions(self):
    options = [
        '--wait_ns=100',
        '--use_cudaDeviceEnablePeerAccess=false',
        '--use_cudaMemcpyDefault=false',
        '--use_cudaMemcpyPeerAsync=true',
    ]
    for opt in options:
      args = self.memcpy_gemm_args + [opt]
      result = subprocess.run(
          args, timeout=self.timeout_s, stdout=subprocess.PIPE)
      self.assertEqual(result.returncode, 0)
      self.ValidateOutput(result.stdout)
    # test sm copy
    result = subprocess.run(
        self.sm_copy_memcpy_args, timeout=self.timeout_s, stdout=subprocess.PIPE)
    self.assertEqual(result.returncode, 0)
    self.ValidateOutput(result.stdout)

  def testFlowModels(self):
    options = ['--flow_model=thread-per-gpu', '--flow_model=event-poll']
    for opt in options:
      args = self.memcpy_gemm_args + [opt]
      result = subprocess.run(
          args, timeout=self.timeout_s, stdout=subprocess.PIPE)
      self.assertEqual(result.returncode, 0)
      self.ValidateOutput(result.stdout)

  def testHighTimeLowTime(self):
    args = self.memcpy_gemm_args + [
        '--gemm', '--N=1024', '--high_time=2', '--low_time=0'
    ]
    result = subprocess.run(
        args, timeout=self.timeout_s, stdout=subprocess.PIPE)
    self.assertEqual(result.returncode, 0)
    self.ValidateOutput(result.stdout)

  # Check for graceful failure when the input/output/compute combination is
  # not supported.
  def testInputComboFailures(self):
    args = self.memcpy_gemm_args + [
        '--gemm',
        '--input_precision=half',
        '--output_precision=single',
        '--compute_precision=double',
    ]
    result = subprocess.run(
        args, timeout=self.timeout_s, stdout=subprocess.PIPE)
    self.assertEqual(result.returncode, 1)


if __name__ == '__main__':
  unittest.main()
