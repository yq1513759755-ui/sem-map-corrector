; SEM raster placement. No COM/ActiveX: works in full AutoCAD on Mac/Windows.
; Generated data: (layer relative-image width height origin u-vector v-vector).
(defun semmap-restore (saved)
  (foreach entry saved (setvar (car entry) (cdr entry))))

(defun semmap-near (a b tolerance)
  (and a b (equal a b tolerance)))

(defun semmap-check-one (entity row / data size)
  (setq data (entget entity) size (cdr (assoc 13 data)))
  (and (= (cdr (assoc 0 data)) "IMAGE")
       (equal (car size) (nth 2 row) 0.01)
       (equal (cadr size) (nth 3 row) 0.01)
       (semmap-near (cdr (assoc 10 data)) (nth 4 row) 0.000001)
       (semmap-near (cdr (assoc 11 data)) (nth 5 row) 0.000000001)
       (semmap-near (cdr (assoc 12 data)) (nth 6 row) 0.000000001)))

(defun semmap-place (row / layer path existing before entity data)
  (setq layer (car row) path (strcat semmap-root (cadr row)))
  (setq existing (ssget "_X" (list '(0 . "IMAGE") (cons 8 layer) '(410 . "Model"))))
  (cond
    (existing
      (if (and (= (sslength existing) 1) (semmap-check-one (ssname existing 0) row))
        (progn (princ (strcat "\nAlready placed: " layer)) T)
        (progn (princ (strcat "\nREVIEW existing image: " layer)) nil)))
    ((not (findfile path)) (princ (strcat "\nMissing image: " path)) nil)
    (T
      (if (not (tblsearch "LAYER" layer))
        (entmake (list '(0 . "LAYER") '(100 . "AcDbSymbolTableRecord")
                       '(100 . "AcDbLayerTableRecord") (cons 2 layer)
                       '(70 . 0) '(62 . 8) '(6 . "Continuous") '(290 . 0))))
      (setq before (entlast))
      (command "_.-IMAGE" "_Attach" (strcat layer "=\"" path "\"")
               "_non" '(0.0 0.0 0.0) 1.0 0.0)
      (setq entity (entlast))
      (if (and entity (not (eq entity before))
               (= (cdr (assoc 0 (entget entity))) "IMAGE"))
        (progn
          (setq data (entget entity))
          (setq data (subst (cons 8 layer) (assoc 8 data) data))
          (setq data (subst (cons 10 (nth 4 row)) (assoc 10 data) data))
          (setq data (subst (cons 11 (nth 5 row)) (assoc 11 data) data))
          (setq data (subst (cons 12 (nth 6 row)) (assoc 12 data) data))
          (if (and (entmod data) (semmap-check-one entity row))
            (progn
              (entupd entity)
              (command "_.DRAWORDER" entity "" "_Back")
              (princ (strcat "\nPlaced: " layer)) T)
            (progn (entdel entity) (princ (strcat "\nFAILED vector verification: " layer)) nil)))
        (progn (princ (strcat "\nFAILED image attach: " layer)) nil)))))

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
          (princ (strcat "\nSEMMAP: " (itoa good) "/" (itoa total) " verified; 1 drawing unit = 1 um. Drawing not saved.")))
        (princ "\nBundle images not found; nothing placed."))))
  (princ))

(defun c:SEMMAP () (semmap-run semmap-data))
(defun c:SEMMAPONE () (semmap-run (if semmap-data (list (car semmap-data)) nil)))

(defun c:SEMMAPCHECK (/ good total selected entity)
  (setq good 0 total (length semmap-data))
  (foreach row semmap-data
    (setq selected (ssget "_X" (list '(0 . "IMAGE") (cons 8 (car row)) '(410 . "Model"))))
    (if (and selected (= (sslength selected) 1) (semmap-check-one (ssname selected 0) row))
      (setq good (1+ good))
      (princ (strcat "\nCHECK FAILED: " (car row)))))
  (princ (strcat "\nSEMMAPCHECK: " (itoa good) "/" (itoa total) " IMAGE transforms verified."))
  (princ))
(princ "\nLoaded SEM map. Run SEMMAPONE first, SEMMAP for all, SEMMAPCHECK to verify.")
(princ)
