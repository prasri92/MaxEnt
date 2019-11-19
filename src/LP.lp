\ Model Zero_atom_detection
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  C0 + C1 + C2 + C3 + C4 + C5 + C6 + C7
Subject To
 R0: C0 = 0.72
 R1: C1 + C3 + C5 + C7 = 0.14
 R2: C2 + C3 + C6 + C7 = 0.1
 R3: C3 + C7 = 0.06
 R4: C4 + C5 + C6 + C7 = 0.18
 R5: C5 + C7 = 0.06
 R6: C6 + C7 = 0.06
 R7: C7 = 0.04
 R8: C0 + C1 + C2 + C3 + C4 + C5 + C6 + C7 = 1
Bounds
 C0 <= 1
 C1 <= 1
 C2 <= 1
 C3 <= 1
 C4 <= 1
 C5 <= 1
 C6 <= 1
 C7 <= 1
End
