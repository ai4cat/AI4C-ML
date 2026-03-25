import os
import re

# function to get replacement rules based on the base name of the structure
# setting N position indices in file names
def get_replacement_rules(base_name):
    if any(x in base_name for x in ['Co_NNNN_Ce_NNNN_2N', 'Co_NNNN_Ce_NNNN_2N1']):
        return {
            0: [3, 5],
            1: [1],
            2: [6],
            3: [0],
            4: [2, 4],
            5: [7]
        }
    elif 'Co_NNNN_Ce_NNNN_1N' in base_name:
        return {
            0: [3, 5],
            1: [0],
            2: [1],
            3: [6],
            4: [4],
            5: [7],
            6: [2]
        }
    elif 'Co_NNNN_Ce_NNNN_di' in base_name:
        return {
            0: [0],  
            1: [1],  
            2: [3],  
            3: [2],  
            4: [4],  
            5: [5],  
            6: [7],  
            7: [6]  
    }
    elif any(x in base_name for x in ['Co_NNNN_N_Ce_NNNN_a', 'Co_NNNN_N_Ce_NNNN_a1']):
        return {
            0: [3, 6],  
            1: [7],  
            2: [8], 
            3: [2, 5],  
            4: [4], 
            5: [0], 
            6: [1]  
    }
    elif 'Co_NNNN_Ce_NNNN_6' in base_name:
        return { 
            0: [3],  
            1: [0],  
            2: [1],  
            3: [4],  
            4: [5],  
            5: [6],  
            6: [7],  
            7: [2]  
    }
    elif 'Co_NNN_Ce_NNNN_b' in base_name:
        return { 
            0: [0],  
            1: [1],  
            2: [2],  
            3: [6],  
            4: [5],  
            5: [3],  
            6: [4] 
    }
    elif 'Co_N_Ce_NNNN_c' in base_name:
        return { 
            0: [0],  
            1: [1],  
            2: [3],  
            3: [4],  
            4: [2],   
    }
    elif 'Co_NNNN_Ce_NNNNN_1N1' in base_name:
        return { 
            0: [0,4],  
            1: [1],  
            2: [2],  
            3: [5],  
            4: [7],
            5: [6],
            6: [3,8],   
    }
    elif 'La_NNNN_Ce_NNNN_0' in base_name:
        return { 
            0: [5],
            1: [2],  
            2: [7],  
            3: [4], 
            4: [0],
            5: [1],
            6: [6], 
            7: [3],  
    }
    elif 'La_NNNNN_Ce_NNNN_d' in base_name:
        return { 
            0: [7],  
            1: [1],  
            2: [0],  
            3: [2],  
            4: [6],
            5: [3],
            6: [8], 
            7: [4,5],  
    }  
    elif 'Co_NNN_Ce_NN_e' in base_name:
        return {
            0: [0],
            1: [1,3],
            2: [2,4],
    }
    elif 'La_NNNN_Ce_NNNN_O6' in base_name:
        return {
            0: [0,4],
            1: [1],
            2: [2],
            3: [4],
            4: [6],
            5: [5],
            6: [3],
    }
    elif 'Co_COOO_Ce_CNO_opt' in base_name:
        return {
            0: [0],
    }
    elif any(x in base_name for x in ['La_NNNNNNNN_Ce_NNNNNNNN', 'La_NNNNNNNN_Ce_NNNNNNNN_O2', 'La_NNNNNNNN_Ce_NNNNNNNN_leaching']):
        return {
            0: [0],
            1: [1],
            2: [11],
            3: [2],
            4: [4],
            5: [5],
            6: [13],
            7: [14],
            8: [3, 8],
            9: [9],
            10: [10],
            11: [6, 12],
            12: [15],
            13: [7],
    }
    # elif 'La_NNNNNN_Ce_NNNNNNN_O2' in base_name:
    #     return {
    #         0: [0],
    #         1: [1],
    #         2: [8],
    #         3: [4],
    #         4: [3],
    #         5: [2],
    #         6: [9],
    #         7: [7],
    #         8: [5, 6],
    #         9: [10],
    #         10: [11],
    #         11: [12],
    # }
    else:
        raise ValueError(f"Unrecognized structure type in filename: {base_name}")
    