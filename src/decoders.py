"""Adapted from BP-OSD code at github.com/viszlai/gbstim"""

import pathlib
import numpy as np
from sinter import CompiledDecoder, Decoder
from stim import DetectorErrorModel

from beliefmatching import detector_error_model_to_check_matrices
from ldpc.bplsd_decoder import BpLsdDecoder
from ldpc.bposd_decoder import BpOsdDecoder

class CompiledBPOSD(CompiledDecoder):

    def __init__(self, check_matrices, decoder):
        self.check_matrices = check_matrices
        self.decoder = decoder

    def decode_shots_bit_packed(self, 
                                bit_packed_detection_event_data: np.ndarray
                               ) -> np.ndarray:
        obs_flip_data = []
        for shot_data in bit_packed_detection_event_data:
            unpacked_data = np.unpackbits(shot_data, bitorder='little', count=self.check_matrices.check_matrix.shape[0])
            pred_errors = self.decoder.decode(unpacked_data)
            obs_pred = (self.check_matrices.observables_matrix @ pred_errors) % 2
            obs_flip_data.append(np.packbits(obs_pred, bitorder='little'))

        return np.array(obs_flip_data)

class BPOSD(Decoder):

    def __init__(self, **kwargs):
        self.decoder_kwargs = kwargs

    def compile_decoder_for_dem(self, 
                                dem: DetectorErrorModel
                               ) -> CompiledDecoder:
        check_matrices = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
        decoder = BpOsdDecoder(check_matrices.check_matrix, channel_probs=check_matrices.priors, **self.decoder_kwargs)
        return CompiledBPOSD(check_matrices, decoder)
    
    def decode_via_files(self, 
                         *, 
                         num_shots: int, 
                         num_dets: int, 
                         num_obs: int, 
                         dem_path: pathlib.Path, 
                         dets_b8_in_path: pathlib.Path, 
                         obs_predictions_b8_out_path: pathlib.Path, 
                         tmp_dir: pathlib.Path
                        ) -> None:
        raise NotImplementedError()
    
class BPLSD(Decoder):
    def __init__(self, **kwargs):
        self.decoder_kwargs = kwargs

    def compile_decoder_for_dem(self, 
                                dem: DetectorErrorModel
                               ) -> CompiledDecoder:
        check_matrices = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
        decoder = BpLsdDecoder(check_matrices.check_matrix, channel_probs=check_matrices.priors, **self.decoder_kwargs)
        return CompiledBPOSD(check_matrices, decoder)

    def decode_via_files(self, 
                         *, 
                         num_shots: int, 
                         num_dets: int, 
                         num_obs: int, 
                         dem_path: pathlib.Path, 
                         dets_b8_in_path: pathlib.Path, 
                         obs_predictions_b8_out_path: pathlib.Path, 
                         tmp_dir: pathlib.Path
                        ) -> None:
        raise NotImplementedError()

# decoder = tesseract_decoder.tesseract.TesseractDecoder(
#     config=tesseract_decoder.tesseract.TesseractConfig(
#         **self.decoder_config | {'det_orders': tesseract_decoder.utils.build_det_orders(
#             dem, num_det_orders=20, det_order_bfs=True, seed=2384753),
#         }
#     ),
# )

DEFAULT_TESSERACT_CONFIG = {
    'pqlimit': 200000,
    'det_beam': 5,
    'beam_climbing': True,
    'no_revisit_dets': True,
}

