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

  def setUp(self):
    self.duration_s = 5
    self.timeout_s = self.duration_s + 15
    exec_path = 'memcpy_gemm'
    self.memcpy_gemm_args = [
        os.path.abspath(exec_path),
        '--duration={}s'.format(self.duration_s),
        # Tests running on forge can only use 1 GPU.
        '--gpus=0',
        '--flows=c0-g0-a0 g0-c0-a1',
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

  def testSGEMM(self):
    args = self.memcpy_gemm_args + [
        '--gemm',
        '--fp_precision=single',
        '--N=2048',
    ]
    result = subprocess.run(
        args, timeout=self.timeout_s, stdout=subprocess.PIPE)
    self.assertEqual(result.returncode, 0)
    self.ValidateOutput(result.stdout)

  def testDGEMM(self):
    args = self.memcpy_gemm_args + [
        '--gemm',
        '--fp_precision=double',
        '--N=2048',
    ]
    result = subprocess.run(
        args, timeout=self.timeout_s, stdout=subprocess.PIPE)
    self.assertEqual(result.returncode, 0)
    self.ValidateOutput(result.stdout)


if __name__ == '__main__':
  unittest.main()
