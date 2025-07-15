from utils import parser
import os

from scripts.eeg import preprocess

import logging

logger = logging.getLogger(__name__)

def main():
    # Set and read arguments
    args = parser.get_parser()

    logger.info('Start preprocessing EEG data')

    id = 'sub-010002'

    # Load raw EEG data
    raw = preprocess.load_data(args, id)

    # Set the VEOG channel as EOG type (not EEG)
    raw.set_channel_types({'VEOG': 'eog'})

    # Set montage
    preprocess.set_montage(args, raw)

    # Set EEG reference
    raw = preprocess.set_eeg_reference(raw)

    # Divide conditions
    raw_eo, raw_ec = preprocess.divide_conditions(raw)

    raw_eo = preprocess.filter_data(args, raw_eo)
    raw_ec = preprocess.filter_data(args, raw_ec)

    # If ICA is requested, apply ICA
    if args.ica:
        logger.info('Applying ICA...')
        raw_eo = preprocess.apply_ica(raw_eo, args, id, condition='EO')
        raw_ec = preprocess.apply_ica(raw_ec, args, id, condition='EC')

    # If plotting is requested, plot the results
    # if args.plot:
    #     logger.info('Plotting results...')
    #     parser.plot_results(raw_eo, raw_ec, args)

    # Save the preprocessed data
    preprocess.save_raw(raw_eo, args.preprocess_path, id, condition='EO')
    preprocess.save_raw(raw_ec, args.preprocess_path, id, condition='EC')

    logger.info('All preprocessing steps completed successfully.')
    
    logger.info('finish')
        

if __name__ == '__main__':
    main()