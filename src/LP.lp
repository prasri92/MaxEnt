\ Model Zero_atom_detection
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  C7 + C9 + C14
Subject To
 R0: C8 + C9 + C10 + C11 + C12 + C13 + C14 + C15 = 0.2
 R1: C4 + C5 + C6 + C7 + C12 + C13 + C14 + C15 = 0.2
 R2: C2 + C3 + C6 + C7 + C10 + C11 + C14 + C15 = 0.22
 R3: C1 + C3 + C5 + C7 + C9 + C11 + C13 + C15 = 0.22
 R4: C0 = 0.48
 R5: C12 + C13 + C14 + C15 = 0.08
 R6: C10 + C11 + C14 + C15 = 0.08
 R7: C9 + C11 + C13 + C15 = 0.08
 R8: C5 + C7 + C13 + C15 = 0.08
 R9: C3 + C7 + C11 + C15 = 0.08
 R10: C0 + C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 + C10 + C11 + C12
   + C13 + C14 + C15 = 1
Bounds
 C0 <= 1
 C1 <= 1
 C2 <= 1
 C3 <= 1
 C4 <= 1
 C5 <= 1
 C6 <= 1
 C7 <= 1
 C8 <= 1
 C9 <= 1
 C10 <= 1
 C11 <= 1
 C12 <= 1
 C13 <= 1
 C14 <= 1
 C15 <= 1
End
