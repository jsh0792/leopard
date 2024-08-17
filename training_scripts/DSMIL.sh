source activate torch

CUDA_VISIBLE_DEVICES="5" python LEOPARD/main.py --config=LEOPARD/config/DSMIL.yaml\
                                                    --start_fold 4\
                                                    --end_fold 5\