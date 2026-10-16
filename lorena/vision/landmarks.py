"""Índices de landmarks do MediaPipe Face Mesh (478 pontos, com íris).

Os índices são ANATÔMICOS (olho esquerdo = olho esquerdo do usuário) quando a
detecção roda no frame não espelhado.
"""

# Cantos ordenados esquerda→direita da IMAGEM (frame não espelhado).
RIGHT_EYE_CORNERS = (33, 133)    # olho direito do usuário: externo, interno
LEFT_EYE_CORNERS = (362, 263)    # olho esquerdo do usuário: interno, externo

# 6 pontos para o EAR: p1 (canto), p2, p3 (pálpebra sup.), p4 (canto), p5, p6 (pálpebra inf.)
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR = (362, 385, 387, 263, 373, 380)

# Pálpebra superior (centro) para a feature de abertura vertical.
RIGHT_UPPER_LID = 159
LEFT_UPPER_LID = 386

# Clusters de 3 pontos para cantos e pálpebras superiores (redução de ruído angular e variância).
RIGHT_EYE_START_CLUSTER = (33, 246, 7)     # Canto externo direito (imagem esquerda)
RIGHT_EYE_END_CLUSTER = (133, 155, 173)    # Canto interno direito (imagem direita)
RIGHT_UPPER_LID_CLUSTER = (159, 160, 158)  # Pálpebra superior direita
LEFT_EYE_START_CLUSTER = (362, 382, 398)   # Canto interno esquerdo (imagem esquerda)
LEFT_EYE_END_CLUSTER = (263, 466, 249)     # Canto externo esquerdo (imagem direita)
LEFT_UPPER_LID_CLUSTER = (386, 385, 387)   # Pálpebra superior esquerda

# Centros de íris (a atribuição a cada olho é verificada por proximidade).
IRIS_CENTERS = (468, 473)
IRIS_RING_A = (469, 470, 471, 472)
IRIS_RING_B = (474, 475, 476, 477)

NUM_LANDMARKS_WITH_IRIS = 478

