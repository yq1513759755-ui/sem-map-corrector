; SEM raster placement. No COM/ActiveX: works in full AutoCAD on Mac/Windows.
; Generated data: (image-name relative-image width height origin u-vector v-vector).
; Images are placed on layer "0" (no per-image layers are created).
(defun semmap-restore (saved)
  (foreach entry saved (setvar (car entry) (cdr entry))))

(defun semmap-near (a b tolerance)
  (and a b (equal a b tolerance)))

(defun semmap-imagedef-filename (entity / data def)
  (setq data (entget entity))
  (if (setq def (cdr (assoc 340 data)))
    (cdr (assoc 1 (entget def)))))

(defun semmap-check-one (entity row / data size)
  (setq data (entget entity) size (cdr (assoc 13 data)))
  (and (= (cdr (assoc 0 data)) "IMAGE")
       (equal (car size) (nth 2 row) 0.01)
       (equal (cadr size) (nth 3 row) 0.01)
       (semmap-near (cdr (assoc 10 data)) (nth 4 row) 0.000001)
       (semmap-near (cdr (assoc 11 data)) (nth 5 row) 0.000000001)
       (semmap-near (cdr (assoc 12 data)) (nth 6 row) 0.000000001)))

(defun semmap-same-file (entity row / fname base)
  (setq fname (semmap-imagedef-filename entity)
        base (vl-filename-base (cadr row)))
  (and fname base
       (wcmatch (strcase fname) (strcat "*" (strcase base) ".*"))))

(defun semmap-place (row / name path existing before entity data)
  (setq name (car row) path (strcat semmap-root (cadr row)))
  (setq existing (ssget "_X" (list '(0 . "IMAGE") '(410 . "Model"))))
  (cond
    ((and existing (semmap-find-same existing row))
      (progn (princ (strcat "\nAlready placed: " name)) T))
    ((and existing (semmap-find-similar existing row))
      (progn (princ (strcat "\nREVIEW existing image: " name)) nil))
    ((not (findfile path)) (princ (strcat "\nMissing image: " path)) nil)
    (T
      (setq before (entlast))
      (command "_.-IMAGE" "_Attach" (strcat name "=\"" path "\"")
               "_non" '(0.0 0.0 0.0) 1.0 0.0)
      (setq entity (entlast))
      (if (and entity (not (eq entity before))
               (= (cdr (assoc 0 (entget entity))) "IMAGE"))
        (progn
          (setq data (entget entity))
          (setq data (subst (cons 8 "0") (assoc 8 data) data))
          (setq data (subst (cons 10 (nth 4 row)) (assoc 10 data) data))
          (setq data (subst (cons 11 (nth 5 row)) (assoc 11 data) data))
          (setq data (subst (cons 12 (nth 6 row)) (assoc 12 data) data))
          (if (and (entmod data) (semmap-check-one entity row))
            (progn
              (entupd entity)
              (command "_.DRAWORDER" entity "" "_Back")
              (princ (strcat "\nPlaced: " name)) T)
            (progn (entdel entity) (princ (strcat "\nFAILED vector verification: " name)) nil)))
        (progn (princ (strcat "\nFAILED image attach: " name)) nil)))))

(defun semmap-find-same (ss row / i entity found)
  (setq i 0 found nil)
  (while (and (< i (sslength ss)) (not found))
    (setq entity (ssname ss i))
    (if (and (semmap-same-file entity row) (semmap-check-one entity row))
      (setq found entity))
    (setq i (1+ i)))
  found)

(defun semmap-find-similar (ss row / i entity found)
  (setq i 0 found nil)
  (while (and (< i (sslength ss)) (not found))
    (setq entity (ssname ss i))
    (if (semmap-same-file entity row)
      (setq found entity))
    (setq i (1+ i)))
  found)

(defun semmap-run (rows / *error* saved total good selected failed)
  (setq saved (mapcar '(lambda (name) (cons name (getvar name))) '("FILEDIA" "OSMODE" "CMDECHO")))
  (defun *error* (message)
    (semmap-restore saved)
    (princ (strcat "\nSEMMAP stopped: " message ". Drawing not saved.")) (princ))
  (cond
    ((/= (getvar "TILEMODE") 1) (princ "\nSwitch to Model space first."))
    ((not rows) (princ "\nNo validated images in this bundle."))
    (T
      (if (not (findfile (strcat semmap-root (cadr (car rows)))))
        (progn
          (setq selected (getfiled "Select this bundle's sem_map.lsp" "" "lsp" 0))
          (if selected (setq semmap-root (strcat (vl-filename-directory selected) "/")))))
      (if (findfile (strcat semmap-root (cadr (car rows))))
        (progn
          (setvar "FILEDIA" 0) (setvar "OSMODE" 0) (setvar "CMDECHO" 0)
          (setq total (length rows) good 0 failed nil)
          (foreach row rows
            (if (not failed)
              (if (semmap-place row) (setq good (1+ good)) (setq failed T))))
          (semmap-restore saved)
          (command "_.REGEN")
          (princ (strcat "\nSEMMAP: " (itoa good) "/" (itoa total) " verified; layer 0 only; 1 drawing unit = 1 um. Drawing not saved.")))
        (princ "\nBundle images not found; nothing placed."))))
  (princ))

(defun c:SEMMAP () (semmap-run semmap-data))
(defun c:SEMMAPONE () (semmap-run (if semmap-data (list (car semmap-data)) nil)))

(defun c:SEMMAPCHECK (/ good total ss entity)
  (setq good 0 total (length semmap-data))
  (setq ss (ssget "_X" (list '(0 . "IMAGE") '(410 . "Model"))))
  (foreach row semmap-data
    (if (and ss (semmap-find-same ss row))
      (setq good (1+ good))
      (princ (strcat "\nCHECK FAILED: " (car row)))))
  (princ (strcat "\nSEMMAPCHECK: " (itoa good) "/" (itoa total) " IMAGE transforms verified."))
  (princ))
(princ "\nLoaded SEM map (layer 0). Run SEMMAPONE first, SEMMAP for all, SEMMAPCHECK to verify.")
(princ)
