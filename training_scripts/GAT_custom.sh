source activate torch

# CUDA_VISIBLE_DEVICES="6" python LEOPARD/main.py --config=LEOPARD/config/GAT_custom.yaml\
#                                                   --start_fold 0\
#                                                   --end_fold 1

# CUDA_VISIBLE_DEVICES="6" python LEOPARD/main.py --config=LEOPARD/config/GAT_custom.yaml\
#                                                   --start_fold 1\
#                                                   --end_fold 2

# CUDA_VISIBLE_DEVICES="6" python LEOPARD/main.py --config=LEOPARD/config/GAT_custom.yaml\
#                                                   --start_fold 2\
#                                                   --end_fold 3

CUDA_VISIBLE_DEVICES="6" python LEOPARD/main.py --config=LEOPARD/config/GAT_custom.yaml\
                                                  --start_fold 3\
                                                  --end_fold 4

CUDA_VISIBLE_DEVICES="6" python LEOPARD/main.py --config=LEOPARD/config/GAT_custom.yaml\
                                                  --start_fold 4\
                                                  --end_fold 5