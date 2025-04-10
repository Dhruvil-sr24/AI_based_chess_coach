# # import sys
# # import numpy as np
# # import mss
# # import torch
# # from PIL import Image
# # from ultralytics import YOLO
# # from stockfish import Stockfish
# # from transformers import pipeline
# # from PyQt5.QtWidgets import QApplication, QWidget
# # from PyQt5.QtCore import Qt, QTimer
# # from PyQt5.QtGui import QPainter, QColor, QFont
# # import pygetwindow as gw


# # class ChessOverlay(QWidget):
# #     def __init__(self, fps=5):
# #         super().__init__(None,
# #             Qt.WindowStaysOnTopHint |
# #             Qt.FramelessWindowHint |
# #             Qt.Tool
# #         )
# #         # Transparent & click-through
# #         self.setAttribute(Qt.WA_TranslucentBackground)
# #         self.setAttribute(Qt.WA_TransparentForMouseEvents)

# #         # Updated region based on confirmed coordinates
# #         self.left, self.top = 660, 230
# #         self.win_width, self.win_height = 520, 520  # 1180-660=520, 750-230=520
# #         self.setGeometry(self.left, self.top, self.win_width, self.win_height)

# #         # Setup mss capture on that region
# #         self.sct = mss.mss()
# #         self.region = {
# #             "left": self.left,
# #             "top": self.top,
# #             "width": self.win_width,
# #             "height": self.win_height
# #         }

# #         # Check for GPU and optimize
# #         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #         print(f"Using device: {self.device}")
        
# #         if self.device.type == 'cuda':
# #             torch.backends.cudnn.benchmark = True
# #             torch.backends.cudnn.deterministic = False

# #         # Load models
# #         self.yolo = YOLO("jsn_best.pt")
# #         self.yolo.to(self.device)  # Explicitly move to GPU
        
# #         # Warm up the model
# #         dummy_input = torch.zeros((1, 3, 640, 640)).to(self.device)
# #         for _ in range(3):
# #             _ = self.yolo.predict(source=dummy_input, verbose=False)
            
# #         self.stockfish = Stockfish(path="stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
        
# #         # Load text generation model only when needed to save memory
# #         self.text_gen = None

# #         # Board tracking
# #         self.prev_board = [["" for _ in range(8)] for _ in range(8)]
# #         self.commentary = "Initializing..."

# #         # Timer for real-time loop
# #         self.timer = QTimer()
# #         self.timer.timeout.connect(self.process_frame)
# #         self.timer.start(int(1000 / fps))

# #         self.show()

# #     def process_frame(self):
# #         try:
# #             windows = gw.getWindowsWithTitle('lichess.org')
# #             if not windows:
# #                 print("Chess window not found!")
# #                 return
            
# #             # Capture & convert to RGB via PIL
# #             sct_img = self.sct.grab(self.region)
# #             pil_img = Image.frombytes('RGB', (sct_img.width, sct_img.height), sct_img.rgb)
# #             gray_img = pil_img.convert('L')
# #             frame = np.array(gray_img)
# #             frame = np.stack((frame,) * 3, axis=-1)  # Convert grayscale to 3 channels

# #             # YOLOv8 detection
# #             results = self.yolo.predict(
# #                 source=frame, 
# #                 verbose=False,
# #                 device=self.device,
# #                 half=True if self.device.type == 'cuda' else False,
# #                 conf=0.25
# #             )
# #             print(f"YOLO results: {results}")

# #             if not results or len(results) == 0:
# #                 print("No results returned from YOLO model")
# #                 return

# #             result = results[0]
# #             if result.boxes is None:
# #                 print("No detections found in this frame")
# #                 return

# #             dets = result.boxes
# #             board_box = None
# #             for i in range(len(dets)):
# #                 d = dets[i]
# #                 if int(d.cls.item()) == 7:
# #                     x1, y1, x2, y2 = map(int, d.xyxy[0].tolist())
# #                     board_box = (x1, y1, x2, y2)
# #                     break

# #             print(f"Board box: {board_box}")
# #             if board_box is None:
# #                 print("No board detected")
# #                 return

# #             x1, y1, x2, y2 = board_box
# #             w, h = x2 - x1, y2 - y1
# #             cw, ch = w / 8, h / 8

# #             mapping = {
# #                 0:"B",1:"K",2:"N",3:"P",4:"Q",5:"R",
# #                 6:"b",8:"k",9:"n",10:"p",11:"q",12:"r"
# #             }
# #             curr = [["" for _ in range(8)] for _ in range(8)]
            
# #             for i in range(len(dets)):
# #                 d = dets[i]
# #                 cls = int(d.cls.item())
# #                 if cls == 7: continue
# #                 xa, ya, xb, yb = d.xyxy[0].tolist()
# #                 cx, cy = (xa+xb)/2, (ya+yb)/2
# #                 col = int((cx - x1) / cw)
# #                 row = int((cy - y1) / ch)
# #                 if 0<=row<8 and 0<=col<8:
# #                     curr[row][col] = mapping.get(cls, "")

# #             move = self.detect_move(self.prev_board, curr)
# #             print(f"Detected move: {move}")
# #             if move:
# #                 fen = self.generate_fen(self.prev_board)
# #                 print(f"FEN: {fen}")
# #                 self.stockfish.set_fen_position(fen)
# #                 self.stockfish.make_moves_from_current_position([move])
# #                 ev = self.stockfish.get_evaluation()
# #                 print(f"Evaluation: {ev}")
# #                 self.commentary = self.generate_simple_commentary(move, ev)
# #                 print(f"Commentary: {self.commentary}")
# #                 self.prev_board = curr
                
# #         except Exception as e:
# #             print(f"Error during processing: {e}")
# #             import traceback
# #             traceback.print_exc()

# #         self.update()

# #     def detect_move(self, prev, curr):
# #         f_sq = t_sq = None
# #         for r in range(8):
# #             for c in range(8):
# #                 if prev[r][c] and not curr[r][c]:
# #                     f_sq = (r, c)
# #                 if not prev[r][c] and curr[r][c]:
# #                     t_sq = (r, c)
# #         if f_sq and t_sq:
# #             f = chr(ord("a")+f_sq[1]) + str(8-f_sq[0])
# #             t = chr(ord("a")+t_sq[1]) + str(8-t_sq[0])
# #             return f+t
# #         return None

# #     def generate_fen(self, board):
# #         rows=[]
# #         for row in board:
# #             empty, s = 0, ""
# #             for cell in row:
# #                 if not cell: empty+=1
# #                 else:
# #                     if empty: s+=str(empty); empty=0
# #                     s+=cell
# #             if empty: s+=str(empty)
# #             rows.append(s)
# #         return "/".join(rows) + " w KQkq - 0 1"

# #     def generate_simple_commentary(self, move, ev):
# #         # Simple commentary without using the heavy transformer model
# #         if isinstance(ev, dict) and 'type' in ev:
# #             if ev['type'] == 'cp':
# #                 score = ev['value'] / 100.0  # Convert centipawns to pawns
# #                 if score > 0:
# #                     return f"Move {move}: White has advantage ({score:.1f} pawns)"
# #                 elif score < 0:
# #                     return f"Move {move}: Black has advantage ({-score:.1f} pawns)"
# #                 else:
# #                     return f"Move {move}: Position is equal"
# #             elif ev['type'] == 'mate':
# #                 mate_in = ev['value']
# #                 if mate_in > 0:
# #                     return f"Move {move}: White has mate in {mate_in}"
# #                 else:
# #                     return f"Move {move}: Black has mate in {-mate_in}"
# #         return f"Move {move} played"

# #     def generate_commentary(self, move, ev):
# #         # Only load and use the text model when explicitly needed
# #         if self.text_gen is None:
# #             self.text_gen = pipeline("text-generation", model="gpt2")
# #         prompt = f"After move {move}, Stockfish evaluates as {ev}. In simple terms, this means"
# #         return self.text_gen(prompt, max_length=50)[0]["generated_text"]

# #     def paintEvent(self, e):
# #         p = QPainter(self)
# #         p.setPen(QColor(0,255,0))
# #         p.setFont(QFont("Consolas",14))
# #         m = 10
# #         bw = 300
# #         bh = self.height() - 2*m
# #         p.fillRect(self.width()-bw-m, m, bw, bh, QColor(0,0,0,150))
# #         y = m+20
# #         for line in self.commentary.split("\n"):
# #             p.drawText(self.width()-bw+m, y, line)
# #             y += 20

# # if __name__ == "__main__":
# #     app = QApplication(sys.argv)
# #     overlay = ChessOverlay(fps=5)
# #     sys.exit(app.exec_())
# import sys
# import numpy as np
# import mss
# import torch
# from PIL import Image
# from ultralytics import YOLO
# from stockfish import Stockfish
# from transformers import pipeline
# from PyQt5.QtWidgets import QApplication, QWidget
# from PyQt5.QtCore import Qt, QTimer
# from PyQt5.QtGui import QPainter, QColor, QFont

# class ChessOverlay(QWidget):
#     def __init__(self, fps=5):
#         super().__init__(None,
#             Qt.WindowStaysOnTopHint |
#             Qt.FramelessWindowHint |
#             Qt.Tool
#         )
#         # Transparent & click-through
#         self.setAttribute(Qt.WA_TranslucentBackground)
#         self.setAttribute(Qt.WA_TransparentForMouseEvents)

#         # --- HARDCODED REGION ---
#         self.left, self.top = 660, 230
#         self.win_width, self.win_height = 520, 520
#         self.setGeometry(self.left, self.top, self.win_width, self.win_height)

#         # Setup mss capture on that region
#         self.sct = mss.mss()
#         self.region = {
#             "left":   self.left,
#             "top":    self.top,
#             "width":  self.win_width,
#             "height": self.win_height
#         }

#         # Check for GPU and optimize
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"Using device: {self.device}")
        
#         if self.device.type == 'cuda':
#             torch.backends.cudnn.benchmark = True
#             torch.backends.cudnn.deterministic = False

#         # Load models
#         self.yolo = YOLO("jsn_best.pt")
#         self.yolo.to(self.device)  # Explicitly move to GPU
        
#         # Warm up the model
#         dummy_input = torch.zeros((1, 3, 640, 640)).to(self.device)
#         for _ in range(3):
#             _ = self.yolo.predict(source=dummy_input, verbose=False)
            
#         self.stockfish = Stockfish(path="stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe")
        
#         # Load text generation model only when needed to save memory
#         self.text_gen = None

#         # Board tracking
#         self.prev_board = [["" for _ in range(8)] for _ in range(8)]
#         self.commentary = "Initializing..."

#         self.consecutive_identical_frames = 0
#         self.stable_board = None

#         # Timer for real-time loop
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.process_frame)
#         self.timer.start(int(1000 / fps))

#         self.show()

#     # def process_frame(self):
#     #     # 1. Capture & convert to grayscale via PIL
#     #     raw = np.array(self.sct.grab(self.region))[:, :, :3]
#     #     pil = Image.fromarray(raw).convert("L")
#     #     img = np.array(pil.convert("RGB"))

#     #     # 2. YOLOv8 detection with GPU optimization
#     #     results = self.yolo.predict(
#     #         source=img, 
#     #         verbose=False,
#     #         device=self.device,
#     #         half=True if self.device.type == 'cuda' else False  # Use FP16 for faster GPU inference
#     #     )
#     #     print(results[0].boxes.cls)
#     #     dets = results[0].boxes
#     #     # print(dets)
#     #     # 3. Find board box (class 7)
#     #     board_box = None
#     #     for d in dets:
#     #         if int(d.cls) == 7:
#     #             x1, y1, x2, y2 = map(int, d.xyxy[0])
#     #             board_box = (x1, y1, x2, y2)
#     #             break
#     #     if board_box is None:
#     #         return

#     #     x1, y1, x2, y2 = board_box
#     #     w, h = x2 - x1, y2 - y1
#     #     cw, ch = w / 8, h / 8

#     #     # 4. Map pieces to 8×8 grid
#     #     mapping = {
#     #         0:"B",1:"K",2:"N",3:"P",4:"Q",5:"R",
#     #         6:"b",8:"k",9:"n",10:"p",11:"q",12:"r"
#     #     }
#     #     curr = [["" for _ in range(8)] for _ in range(8)]
#     #     for d in dets:
#     #         cls = int(d.cls)
#     #         if cls == 7: continue
#     #         xa, ya, xb, yb = d.xyxy[0]
#     #         cx, cy = (xa+xb)/2, (ya+yb)/2
#     #         col = int((cx - x1) / cw)
#     #         row = int((cy - y1) / ch)
#     #         if 0<=row<8 and 0<=col<8:
#     #             curr[row][col] = mapping.get(cls, "")

#     #     # 5. Detect move
#     #     move = self.detect_move(self.prev_board, curr)
#     #     if move:
#     #         fen = self.generate_fen(self.prev_board)
#     #         self.stockfish.set_fen_position(fen)
#     #         self.stockfish.make_moves_from_current_position([move])
#     #         ev = self.stockfish.get_evaluation()
#     #         self.commentary = self.generate_commentary(move, ev)
#     #         self.prev_board = curr

#     #     self.update()
#     def process_frame(self):
#         try:
#             # 1. Capture & convert to grayscale via PIL
#             raw = np.array(self.sct.grab(self.region))[:, :, :3]
#             pil = Image.fromarray(raw).convert("L")
#             img = np.array(pil.convert("RGB"))

#             # 2. YOLOv8 detection with GPU optimization
#             results = self.yolo.predict(
#                 source=img, 
#                 verbose=False,
#                 device=self.device,
#                 half=True if self.device.type == 'cuda' else False
#             )
            
#             if len(results) == 0:
#                 print("No detection results")
#                 return
                
#             # Debug: Print the shape and content of detection results
#             print(f"Detected {len(results[0].boxes)} objects")
            
#             dets = results[0].boxes
#             if len(dets) == 0:
#                 print("No boxes detected")
#                 return
                
#             # 3. Find board box (class 7)
#             board_box = None
#             for i in range(len(dets)):
#                 d = dets[i]
#                 cls_val = int(d.cls.item())  # Convert tensor to int correctly
#                 if cls_val == 7:
#                     x1, y1, x2, y2 = map(int, d.xyxy[0].tolist())
#                     board_box = (x1, y1, x2, y2)
#                     break
                    
#             if board_box is None:
#                 print("No chessboard detected")
#                 return

#             x1, y1, x2, y2 = board_box
#             w, h = x2 - x1, y2 - y1
#             cw, ch = w / 8, h / 8

#             # 4. Map pieces to 8×8 grid
#             mapping = {
#                 0:"B",1:"K",2:"N",3:"P",4:"Q",5:"R",
#                 6:"b",8:"k",9:"n",10:"p",11:"q",12:"r"
#             }
#             curr = [["" for _ in range(8)] for _ in range(8)]
            
#             pieces_detected = 0
#             for i in range(len(dets)):
#                 d = dets[i]
#                 cls_val = int(d.cls.item())  # Correctly convert tensor to int
#                 if cls_val == 7: 
#                     continue
                    
#                 xa, ya, xb, yb = d.xyxy[0].tolist()
#                 cx, cy = (xa+xb)/2, (ya+yb)/2
#                 col = int((cx - x1) / cw)
#                 row = int((cy - y1) / ch)
#                 if 0 <= row < 8 and 0 <= col < 8:
#                     curr[row][col] = mapping.get(cls_val, "")
#                     pieces_detected += 1
                    
#             print(f"Detected {pieces_detected} chess pieces")
#             board_str = "\n".join(["".join(row) for row in curr])
#             if self.stable_board == board_str:
#                 self.consecutive_identical_frames += 1
#                 if self.consecutive_identical_frames >= 3:  # Only process after 3 identical frames
#                     # Process move detection here
#                     move = self.detect_move(self.prev_board, curr)
#                     if move:
#                         # Handle move as before
#                         self.prev_board = [row[:] for row in curr]
#             else:
#                 self.consecutive_identical_frames = 0
#                 self.stable_board = board_str        
#             # Print current board state for debugging
#             board_str = "\n".join([" ".join([p if p else "." for p in row]) for row in curr])
#             print(f"Current board state:\n{board_str}")

#             # 5. Detect move
#             move = self.detect_move(self.prev_board, curr)
#             if move:
#                 print(f"Move detected: {move}")
#                 fen = self.generate_fen(self.prev_board)
#                 print(f"FEN: {fen}")
#                 try:
#                     self.stockfish.set_fen_position(fen)
#                     self.stockfish.make_moves_from_current_position([move])
#                     ev = self.stockfish.get_evaluation()
#                     self.commentary = self.generate_simple_commentary(move, ev)
#                     self.prev_board = [row[:] for row in curr]  # Deep copy
#                 except Exception as e:
#                     print(f"Stockfish error: {e}")
#                     self.commentary = f"Move {move} detected but couldn't analyze"
#             else:
#                 print("No move detected")
                
#         except Exception as e:
#             print(f"Error in process_frame: {e}")
#             import traceback
#             traceback.print_exc()

#         self.update()

#     # def detect_move(self, prev, curr):
#     #     f_sq = t_sq = None
#     #     for r in range(8):
#     #         for c in range(8):
#     #             if prev[r][c] and not curr[r][c]:
#     #                 f_sq = (r, c)
#     #             if not prev[r][c] and curr[r][c]:
#     #                 t_sq = (r, c)
#     #     if f_sq and t_sq:
#     #         f = chr(ord("a")+f_sq[1]) + str(8-f_sq[0])
#     #         t = chr(ord("a")+t_sq[1]) + str(8-t_sq[0])
#     #         return f+t
#     #     return None
#     def detect_move(self, prev, curr):
#         """Detect moves by comparing previous and current board states"""
#         differences = []
        
#         # Print differences for debugging
#         print("Board differences:")
#         for r in range(8):
#             for c in range(8):
#                 if prev[r][c] != curr[r][c]:
#                     from_sq = f"{chr(ord('a')+c)}{8-r}" if prev[r][c] else None
#                     to_sq = f"{chr(ord('a')+c)}{8-r}" if curr[r][c] else None
#                     print(f"  Position ({r},{c}): {prev[r][c] or '.'} -> {curr[r][c] or '.'}")
#                     differences.append((r, c, prev[r][c], curr[r][c]))
        
#         if len(differences) == 0:
#             return None
        
#         # Case 1: Simple move (one piece disappears, another appears)
#         from_sqs = [(r, c) for r, c, p, _ in differences if p and not curr[r][c]]
#         to_sqs = [(r, c) for r, c, p, cp in differences if not p and cp]
        
#         if len(from_sqs) == 1 and len(to_sqs) == 1:
#             fr, fc = from_sqs[0]
#             tr, tc = to_sqs[0]
#             f = chr(ord("a")+fc) + str(8-fr)
#             t = chr(ord("a")+tc) + str(8-tr)
#             return f+t
        
#         # Case 2: Capture (one piece disappears, another changes)
#         # This is a simplification and may not catch all cases
#         if len(differences) == 2:
#             piece_moves = [d for d in differences if d[2] and d[3]]
#             piece_vanishes = [d for d in differences if d[2] and not d[3]]
            
#             if len(piece_vanishes) == 1 and len(piece_moves) == 0:
#                 r, c, _, _ = piece_vanishes[0]
#                 f = chr(ord("a")+c) + str(8-r)
                
#                 # Find the closest new piece as destination
#                 min_dist = float('inf')
#                 best_tr, best_tc = None, None
#                 for tr, tc, prev_p, curr_p in differences:
#                     if not prev_p and curr_p:  # A new piece appeared
#                         dist = abs(r - tr) + abs(c - tc)
#                         if dist < min_dist:
#                             min_dist = dist
#                             best_tr, best_tc = tr, tc
                
#                 if best_tr is not None:
#                     t = chr(ord("a")+best_tc) + str(8-best_tr)
#                     return f+t
        
#         print("Could not determine a clear move from the differences")
#         return None
#     def generate_fen(self, board):
#         rows=[]
#         for row in board:
#             empty, s = 0, ""
#             for cell in row:
#                 if not cell: empty+=1
#                 else:
#                     if empty: s+=str(empty); empty=0
#                     s+=cell
#             if empty: s+=str(empty)
#             rows.append(s)
#         return "/".join(rows) + " w KQkq - 0 1"

#     def generate_simple_commentary(self, move, ev):
#         # Simple commentary without using the heavy transformer model
#         if isinstance(ev, dict) and 'type' in ev:
#             if ev['type'] == 'cp':
#                 score = ev['value'] / 100.0  # Convert centipawns to pawns
#                 if score > 0:
#                     return f"Move {move}: White has advantage ({score:.1f} pawns)"
#                 elif score < 0:
#                     return f"Move {move}: Black has advantage ({-score:.1f} pawns)"
#                 else:
#                     return f"Move {move}: Position is equal"
#             elif ev['type'] == 'mate':
#                 mate_in = ev['value']
#                 if mate_in > 0:
#                     return f"Move {move}: White has mate in {mate_in}"
#                 else:
#                     return f"Move {move}: Black has mate in {-mate_in}"
#         return f"Move {move} played"

#     def generate_commentary(self, move, ev):
#         # Only load and use the text model when explicitly needed
#         if self.text_gen is None:
#             self.text_gen = pipeline("text-generation", model="gpt2")
#         prompt = f"After move {move}, Stockfish evaluates as {ev}. In simple terms, this means"
#         return self.text_gen(prompt, max_length=50)[0]["generated_text"]

#     def paintEvent(self, e):
#         p = QPainter(self)
#         p.setPen(QColor(0, 255, 0))
#         p.setFont(QFont("Consolas", 14))
#         m = 10
#         bw = 300; bh = self.height() - 2 * m
#         # semi‑opaque box
#         p.fillRect(self.width() - bw - m, m, bw, bh, QColor(0, 0, 0, 150))
#         y = m + 20
#         for line in self.commentary.split("\n"):
#             p.drawText(self.width() - bw + m, y, line)
#             y += 20

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     overlay = ChessOverlay(fps=5)
#     sys.exit(app.exec_())
import sys
import numpy as np
import mss
import torch
from PIL import Image
from ultralytics import YOLO
from stockfish import Stockfish
from transformers import pipeline
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QFont

class ChessOverlay(QWidget):
    def __init__(self, fps=5):
        super().__init__(None,
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.Tool
        )
        # Transparent & click-through
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        # Expand window to include side panel for commentary
        self.left, self.top = 660, 230
        self.win_width, self.win_height = 820, 520  # Added 300px width for side panel
        self.board_width = 520  # Original board width
        self.setGeometry(self.left, self.top, self.win_width, self.win_height)

        # Setup mss capture only on the board region (not including the side panel)
        self.sct = mss.mss()
        self.region = {
            "left":   self.left,
            "top":    self.top,
            "width":  self.board_width,  # Only capture the actual board area
            "height": self.win_height
        }

        # Check for GPU and optimize
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # Load models
        self.yolo = YOLO("jsn_best.pt")
        self.yolo.to(self.device)  # Explicitly move to GPU
        
        # Warm up the model
        dummy_input = torch.zeros((1, 3, 640, 640)).to(self.device)
        for _ in range(3):
            _ = self.yolo.predict(source=dummy_input, verbose=False)
            
        self.stockfish = Stockfish(path="stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe")
        
        # Load text generation model only when needed to save memory
        self.text_gen = None

        # Board tracking
        self.prev_board = [["" for _ in range(8)] for _ in range(8)]
        self.commentary = "Initializing..."
        self.move_history = []

        # Timer for real-time loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(int(1000 / fps))

        self.show()

    def process_frame(self):
        try:
            # 1. Capture & convert to grayscale via PIL
            raw = np.array(self.sct.grab(self.region))[:, :, :3]
            pil = Image.fromarray(raw).convert("L")
            img = np.array(pil.convert("RGB"))

            # 2. YOLOv8 detection with GPU optimization
            results = self.yolo.predict(
                source=img, 
                verbose=False,
                device=self.device,
                half=True if self.device.type == 'cuda' else False
            )
            
            if len(results) == 0:
                print("No detection results")
                return
                
            # Debug: Print the shape and content of detection results
            print(f"Detected {len(results[0].boxes)} objects")
            
            dets = results[0].boxes
            if len(dets) == 0:
                print("No boxes detected")
                return
                
            # 3. Find board box (class 7)
            board_box = None
            for i in range(len(dets)):
                d = dets[i]
                cls_val = int(d.cls.item())  # Convert tensor to int correctly
                if cls_val == 7:
                    x1, y1, x2, y2 = map(int, d.xyxy[0].tolist())
                    board_box = (x1, y1, x2, y2)
                    break
                    
            if board_box is None:
                print("No chessboard detected")
                return

            x1, y1, x2, y2 = board_box
            w, h = x2 - x1, y2 - y1
            cw, ch = w / 8, h / 8

            # 4. Map pieces to 8×8 grid
            mapping = {
                0:"B",1:"K",2:"N",3:"P",4:"Q",5:"R",
                6:"b",8:"k",9:"n",10:"p",11:"q",12:"r"
            }
            curr = [["" for _ in range(8)] for _ in range(8)]
            
            pieces_detected = 0
            for i in range(len(dets)):
                d = dets[i]
                cls_val = int(d.cls.item())  # Correctly convert tensor to int
                if cls_val == 7: 
                    continue
                    
                xa, ya, xb, yb = d.xyxy[0].tolist()
                cx, cy = (xa+xb)/2, (ya+yb)/2
                col = int((cx - x1) / cw)
                row = int((cy - y1) / ch)
                if 0 <= row < 8 and 0 <= col < 8:
                    curr[row][col] = mapping.get(cls_val, "")
                    pieces_detected += 1
                    
            print(f"Detected {pieces_detected} chess pieces")
            
            # Validate the board state (remove duplicate kings, etc.)
            curr = self.validate_board_state(curr)
                    
            # Print current board state for debugging
            board_str = "\n".join([" ".join([p if p else "." for p in row]) for row in curr])
            print(f"Current board state:\n{board_str}")

            # Initialize the previous board on first detection with many pieces
            if self.is_empty_board(self.prev_board) and pieces_detected > 20:
                print("Initial board position detected - saving as reference")
                self.prev_board = [row[:] for row in curr]  # Deep copy
                self.commentary = "Board position detected\nWaiting for moves..."
                self.update()
                return

            # 5. Detect move only if we have a previous non-empty board
            if not self.is_empty_board(self.prev_board):
                move = self.detect_move(self.prev_board, curr)
                if move:
                    print(f"Move detected: {move}")
                    fen = self.generate_fen(self.prev_board)
                    print(f"FEN: {fen}")
                    try:
                        self.stockfish.set_fen_position(fen)
                        self.stockfish.make_moves_from_current_position([move])
                        ev = self.stockfish.get_evaluation()
                        self.move_history.append(move)
                        move_num = len(self.move_history)
                        self.commentary = f"Move #{move_num}: {move}\n" + self.generate_simple_commentary(move, ev)
                        self.prev_board = [row[:] for row in curr]  # Deep copy
                    except Exception as e:
                        print(f"Stockfish error: {e}")
                        self.commentary = f"Move {move} detected but couldn't analyze"
            else:
                print("No previous board state to compare with")
                
        except Exception as e:
            print(f"Error in process_frame: {e}")
            import traceback
            traceback.print_exc()

        self.update()

    def is_empty_board(self, board):
        """Check if a board is empty or nearly empty"""
        piece_count = 0
        for row in board:
            for cell in row:
                if cell:
                    piece_count += 1
        return piece_count < 5  # Consider empty if fewer than 5 pieces

    def validate_board_state(self, board):
        """Validate and clean up the board state"""
        white_king_count = 0
        black_king_count = 0
        
        for r in range(8):
            for c in range(8):
                if board[r][c] == 'K':
                    white_king_count += 1
                    if white_king_count > 1:
                        print(f"Multiple white kings detected! Removing extra at {r},{c}")
                        board[r][c] = ''
                elif board[r][c] == 'k':
                    black_king_count += 1
                    if black_king_count > 1:
                        print(f"Multiple black kings detected! Removing extra at {r},{c}")
                        board[r][c] = ''
        
        return board

    def detect_move(self, prev, curr):
        """Detect moves by comparing previous and current board states"""
        from_squares = []
        to_squares = []
        
        # Collect differences
        for r in range(8):
            for c in range(8):
                if prev[r][c] and not curr[r][c]:  # Piece disappeared
                    from_squares.append((r, c, prev[r][c]))
                    print(f"Piece {prev[r][c]} disappeared from {chr(ord('a')+c)}{8-r}")
                elif not prev[r][c] and curr[r][c]:  # Piece appeared
                    to_squares.append((r, c, curr[r][c]))
                    print(f"Piece {curr[r][c]} appeared at {chr(ord('a')+c)}{8-r}")
        
        if len(from_squares) == 1 and len(to_squares) == 1:
            fr, fc, piece_from = from_squares[0]
            tr, tc, piece_to = to_squares[0]
            
            # Check if same color piece moved (not a capture of different color)
            same_color = (piece_from.isupper() and piece_to.isupper()) or \
                         (piece_from.islower() and piece_to.islower())
            
            if same_color and piece_from.upper() == piece_to.upper():
                f = chr(ord("a")+fc) + str(8-fr)
                t = chr(ord("a")+tc) + str(8-tr)
                return f+t
            elif not same_color:  # Capture
                f = chr(ord("a")+fc) + str(8-fr)
                t = chr(ord("a")+tc) + str(8-tr)
                return f+t
        
        # Special case - castling
        if len(from_squares) == 2 and len(to_squares) == 2:
            king_from = None
            rook_from = None
            king_to = None
            rook_to = None
            
            for r, c, p in from_squares:
                if p.upper() == 'K':
                    king_from = (r, c)
                elif p.upper() == 'R':
                    rook_from = (r, c)
                    
            for r, c, p in to_squares:
                if p.upper() == 'K':
                    king_to = (r, c)
                elif p.upper() == 'R':
                    rook_to = (r, c)
            
            if king_from and king_to and rook_from and rook_to:
                # Just report king's move for castling
                fr, fc = king_from
                tr, tc = king_to
                f = chr(ord("a")+fc) + str(8-fr)
                t = chr(ord("a")+tc) + str(8-tr)
                return f+t
        
        print("Could not determine a clear move from the differences")
        return None

    def generate_fen(self, board):
        rows=[]
        for row in board:
            empty, s = 0, ""
            for cell in row:
                if not cell: empty+=1
                else:
                    if empty: s+=str(empty); empty=0
                    s+=cell
            if empty: s+=str(empty)
            rows.append(s)
        return "/".join(rows) + " w KQkq - 0 1"

    def generate_simple_commentary(self, move, ev):
        # Simple commentary without using the heavy transformer model
        if isinstance(ev, dict) and 'type' in ev:
            if ev['type'] == 'cp':
                score = ev['value'] / 100.0  # Convert centipawns to pawns
                if score > 0:
                    return f"White has advantage ({score:.1f} pawns)"
                elif score < 0:
                    return f"Black has advantage ({-score:.1f} pawns)"
                else:
                    return f"Position is equal"
            elif ev['type'] == 'mate':
                mate_in = ev['value']
                if mate_in > 0:
                    return f"White has mate in {mate_in}"
                else:
                    return f"Black has mate in {-mate_in}"
        return f"Move evaluation unavailable"

    def generate_commentary(self, move, ev):
        # Only load and use the text model when explicitly needed
        if self.text_gen is None:
            self.text_gen = pipeline("text-generation", model="gpt2")
        prompt = f"After move {move}, Stockfish evaluates as {ev}. In simple terms, this means"
        return self.text_gen(prompt, max_length=50)[0]["generated_text"]

    def paintEvent(self, e):
        p = QPainter(self)
        
        # Draw the side panel on the right side of the window
        panel_x = self.board_width  # Start panel at board's right edge
        panel_width = self.win_width - self.board_width
        
        # Draw a semi-transparent background for the panel
        p.fillRect(panel_x, 0, panel_width, self.height(), QColor(0, 0, 0, 180))
        
        # Draw divider line
        p.setPen(QColor(100, 100, 100))
        p.drawLine(panel_x, 0, panel_x, self.height())
        
        # Draw title
        p.setPen(QColor(200, 200, 200))
        p.setFont(QFont("Arial", 14, QFont.Bold))
        p.drawText(panel_x + 10, 30, "Chess Analysis")
        
        # Draw horizontal separator
        p.setPen(QColor(100, 100, 100))
        p.drawLine(panel_x + 10, 40, panel_x + panel_width - 10, 40)
        
        # Draw commentary
        p.setPen(QColor(0, 255, 0))
        p.setFont(QFont("Consolas", 12))
        y = 70
        
        # Display the commentary with wrapping
        line_height = 20
        max_line_width = panel_width - 20
        font_metrics = p.fontMetrics()
        
        for paragraph in self.commentary.split("\n"):
            words = paragraph.split()
            current_line = ""
            
            for word in words:
                test_line = current_line + word + " "
                if font_metrics.horizontalAdvance(test_line) <= max_line_width:
                    current_line = test_line
                else:
                    p.drawText(panel_x + 10, y, current_line)
                    y += line_height
                    current_line = word + " "
            
            if current_line:
                p.drawText(panel_x + 10, y, current_line)
                y += line_height
            
            # Add extra space between paragraphs
            y += 5
        
        # Draw move history section if we have moves
        if self.move_history:
            y += 20
            p.setPen(QColor(200, 200, 200))
            p.setFont(QFont("Arial", 12, QFont.Bold))
            p.drawText(panel_x + 10, y, "Move History")
            
            p.setPen(QColor(100, 100, 100))
            p.drawLine(panel_x + 10, y + 10, panel_x + panel_width - 10, y + 10)
            
            y += 30
            p.setPen(QColor(255, 255, 255))
            p.setFont(QFont("Consolas", 11))
            
            for i, move in enumerate(self.move_history):
                move_text = f"{i+1}. {move}"
                p.drawText(panel_x + 10, y, move_text)
                y += line_height
                
                # Only show the last 10 moves if there are many
                if i >= 9 and len(self.move_history) > 10:
                    p.drawText(panel_x + 10, y, "...")
                    break

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = ChessOverlay(fps=5)
    sys.exit(app.exec_())