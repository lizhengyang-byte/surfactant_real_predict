from smiles_to_features_pharmhgt import smiles_to_features_pharmhgt

vec = smiles_to_features_pharmhgt("CCO")
print(vec.shape)  # (522,)
print(vec)  # [0. 0. 0. ... 0. 0. 0.]