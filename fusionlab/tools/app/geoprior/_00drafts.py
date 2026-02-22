    # def _build_ui(self) -> None: 
    #     root = QWidget(self)
    #     self.setCentralWidget(root)

    #     layout = QVBoxLayout(root)
        
    #     self._update_splash(42, "Building top toolbar…") 
        
    #     # --- Top row: [Select CSV…] [City / Dataset] [Dry run] [Mode] [Quit] ---
    #     top = QHBoxLayout()
    
        
    #     self.select_csv_btn = QPushButton("Open dataset…")
    #     top.addWidget(self.select_csv_btn)

    #     label = QLabel("City / Dataset:")
    #     # label.setStyleSheet("font-weight: 600;")
    #     top.addWidget(label)
       
    #     self.city_edit = QLineEdit()
    #     self.city_edit.setPlaceholderText("e.g. nansha")
    #     top.addWidget(self.city_edit, 1)
        
    #     self.city_edit.setStyleSheet(
    #         """
    #         QLineEdit#cityDatasetEdit {
    #             background-color: #fff9f0;
    #             border: 1px solid #f0b96a;
    #             border-radius: 8px;
    #             padding: 3px 8px;
    #             font-weight: 600;
    #         }
    #         QLineEdit#cityDatasetEdit:focus {
    #             border: 1px solid #e8902f;
    #             background-color: #fff3e0;
    #         }
    #         """
    #     )

    #     # stretch between dataset and right-side controls
    #     top.addStretch(1)

    #     # Dry-run checkbox (logic will be added later)
    #     self.chk_dry_run = QCheckBox("Dry run")
    #     self.chk_dry_run.setToolTip(
    #         "Prepare configuration and log actions\n"
    #         " without actually running Stage-1 / Stage-2."
    #     )
    #     top.addWidget(self.chk_dry_run)
        
    #     # Global Stop button – only visible while a job is running
    #     self.btn_stop = QPushButton("Stop")
    #     self.btn_stop.setVisible(False)
    #     self.btn_stop.setEnabled(False)
    #     self.btn_stop.setObjectName("stopButton")
    #     self.btn_stop.setCursor(Qt.PointingHandCursor)
    #     top.addWidget(self.btn_stop)
        
    #     # --- NEW: base + pulse styles for the Stop button ---
    #     self._stop_base_style = """
    #     QPushButton#stopButton {
    #         background-color: #d9534f;   /* red */
    #         color: white;
    #         font-weight: 600;
    #         border-radius: 10px;
    #         padding: 4px 16px;
    #     }
    #     QPushButton#stopButton:hover {
    #         background-color: #c9302c;
    #     }
    #     QPushButton#stopButton:disabled {
    #         background-color: #a0a0a0;
    #         color: #eeeeee;
    #     }
    #     """
        
    #     # Slightly lighter / stronger red for the "pulse" frame
    #     self._stop_pulse_style = """
    #     QPushButton#stopButton {
    #         background-color: #ff6f6f;
    #         color: white;
    #         font-weight: 700;
    #         border-radius: 10px;
    #         padding: 4px 16px;
    #     }
    #     """
        
    #     # Apply the base style once
    #     self.btn_stop.setStyleSheet(self._stop_base_style)
        
    #     # Timer driving the pulse effect
    #     self._stop_pulse_timer = QTimer(self)
    #     self._stop_pulse_timer.setInterval(450)  # ms between frames
    #     self._stop_pulse_state = False

    #     # Mode indicator button (updated when tab changes)
    #     self.mode_btn = QPushButton("Mode: Train")
    #     self.mode_btn.setEnabled(False)
    #     self.mode_btn.setFlat(True)
    #     top.addWidget(self.mode_btn)
        
    #     layout.addLayout(top)
    #     # --- Tabs row: Train / Tune / Inference / Transferability ---
    #     self._update_splash(45, "Building tabs…")
        
    #     self.tabs = QTabWidget()
    #     self._init_tabs()
    #     layout.addWidget(self.tabs, 6) # 1

    #     # --- Status line ---
    #     status_row = QHBoxLayout()

    #     self.status_label = QLabel("? Idle")
    #     self.status_label.setStyleSheet(f"color:{PRIMARY};")
    #     status_row.addWidget(self.status_label, 1)  # takes the left side
        
    #     status_row.addStretch(1)  # push timer fully to the right
        
    #     # digital run timer (black background, green digits)
    #     self.run_timer = RunClockTimer(self)
    #     self.run_timer.reset()
    #     self.run_timer.stop()
    #     self.run_timer.setVisible(False)
    #     status_row.addWidget(self.run_timer, 0)
        
    #     layout.addLayout(status_row)
        
    #     # --- Log widget ---
    #     self.log_widget = QPlainTextEdit()
    #     self.log_widget.setObjectName("logWidget")
    #     self.log_widget.setReadOnly(True)
    #     self.log_widget.setSizePolicy(
    #         QSizePolicy.Expanding,
    #         QSizePolicy.Expanding,
    #     )
    #     self.log_widget.setMinimumHeight(200)
    #     layout.addWidget(self.log_widget, 1)

    #     # --- Progress row ---
    #     prog = QHBoxLayout()

    #     self.progress_label = QLabel("")
    #     self.progress_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    #     self.progress_label.setSizePolicy(
    #         QSizePolicy.Minimum,
    #         QSizePolicy.Fixed,
    #     )
    #     prog.addWidget(self.progress_label, 0)

    #     self.progress_bar = QProgressBar()
    #     self.progress_bar.setMinimumHeight(18)
    #     self.progress_bar.setTextVisible(False)
    #     self.progress_bar.setSizePolicy(
    #         QSizePolicy.Expanding,
    #         QSizePolicy.Fixed,
    #     )
    #     prog.addWidget(self.progress_bar, 1)

    #     self.percent_label = QLabel("0 %")
    #     self.percent_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    #     self.percent_label.setSizePolicy(
    #         QSizePolicy.Minimum,
    #         QSizePolicy.Fixed,
    #     )
    #     prog.addWidget(self.percent_label, 0)

    #     layout.addLayout(prog)
    
    
    # def _on_tab_changed(self, index: int) -> None:
    #     """
    #     Update the Mode indicator and toggle the log visibility
    #     when the active tab changes.
    #     """
    #     self._update_mode_button(index)

    #     if not hasattr(self, "log_widget"):
    #         return
        
    #     data_idx = getattr(self, "_data_tab_index", -1)
    #     results_idx = getattr(self, "_results_tab_index", -1)
    #     tools_idx   = getattr(self, "_tools_tab_index", -1)

    #     # Train / Tune / Inference / Transfer → console visible
    #     # Results → console hidden
    #     # Tools   → hidden by default; individual tools can override.
    #     if index == results_idx:
    #         self.set_console_visible(False)
    #     elif index == tools_idx:
    #         self.set_console_visible(False)   # default for Tools
    #     elif index ==data_idx:
    #         self.set_console_visible(False)
    #     else:
    #         self.set_console_visible(True)
            

    #     if index in (results_idx, tools_idx, data_idx):
    #         self.set_console_visible(False)
    #     else:
    #         self.set_console_visible(True)
            