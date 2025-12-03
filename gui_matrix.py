import FreeSimpleGUI as sg
import numpy as np
from tkinter import Widget

def prnt_num(num):
    return f"{num:06.01f}"

class windows_control:
    def __init__(self):
        self.intrinsic = np.array([
            [1,0,0], 
            [0,1,0], 
            [0,0,1]], dtype = np.float64)
        self.extrinsic = np.array([
            [960,0,960], 
            [0,-540,1080], 
            [0,0,1]], dtype = np.float64)
        self.selected = None

        self.layout = [
            [self.create_slider_layout()],
            [self.create_intrinsic_matrix(), sg.Push(), self.create_extrinsic_matrix()],
            [self.create_cases_selector(), sg.Push() , self.create_control_buttons()]
        ]
        self.window = sg.Window("SLIDER", self.layout, finalize=True)
        self.update_all_buttons()
        self.window.hide()
        self.previous_num = 0.0

    def update_all_buttons(self):
        for x in [0,1,2]:
            for y in [0,1,2]:
                self.window[f"choose_I{x}{y}"].update(prnt_num(self.intrinsic[x][y]))
                self.window[f"choose_E{x}{y}"].update(prnt_num(self.extrinsic[x][y]))

    def create_intrinsic_matrix(self):
        return sg.Frame("Intrinsic Matrix",[
            [sg.Button("NONE", key="choose_I00"),sg.Button("NONE", key="choose_I01"),sg.Button("NONE", key="choose_I02")],
            [sg.Button("NONE", key="choose_I10"),sg.Button("NONE", key="choose_I11"),sg.Button("NONE", key="choose_I12")],
            [sg.Button("NONE", key="choose_I20"),sg.Button("NONE", key="choose_I21"),sg.Button("NONE", key="choose_I22")]
        ], title_location=sg.TITLE_LOCATION_TOP, key="INTRINSIC")

    def create_extrinsic_matrix(self):
        return sg.Frame("Extrinsic Matrix",[
            [sg.Button("NONE", key="choose_E00"),sg.Button("NONE", key="choose_E01"),sg.Button("NONE", key="choose_E02")],
            [sg.Button("NONE", key="choose_E10"),sg.Button("NONE", key="choose_E11"),sg.Button("NONE", key="choose_E12")],
            [sg.Button("NONE", key="choose_E20"),sg.Button("NONE", key="choose_E21"),sg.Button("NONE", key="choose_E22")]
        ], title_location=sg.TITLE_LOCATION_TOP, key="EXTRINSIC")

    def create_slider_layout(self):
        return sg.Frame("Value Adjust", [
            [ sg.Button("-1", key="btn_10"), sg.Button("-0.1", key="btn_1"),  sg.Button("-0.01", key="btn_2"),   sg.Button("-0.001", key="btn_3"),  
            sg.Button("+0.001", key="btn_4"), sg.Button("+0.01", key="btn_5"), sg.Button("+0.1", key="btn_6"), sg.Button("+1", key="btn_11")],
             [sg.Slider((-1080, 1080), (0), (0.01), orientation='h', expand_x=True, key="slider", enable_events=True, disable_number_display=True), 
              sg.Input("0.0", key="input_text", size=(6,20)), sg.Ok(key="input_ok")
              ]
        ], title_location=sg.TITLE_LOCATION_BOTTOM, key="SLIDER")

    def create_cases_selector(self):
        return sg.Frame("Instrinsic specifics", [
            [sg.Button("Fx", key="sel_fx"), sg.Button("Fy", key="sel_fy"), sg.Button("Cx", key="sel_cx"), sg.Button("Cy", key="sel_cy")]
        ], title_location= sg.TITLE_LOCATION_TOP)

    def create_control_buttons(self):
        return sg.Frame("Control Player", [
            [sg.Button("<<", key="con_prev"), sg.Button("||", key="con_pause"), sg.Button(">", key="con_play"), sg.Button(">>", key="con_next")]
        ], title_location= sg.TITLE_LOCATION_TOP)

    def value_from_matrix(self, event, update = False, value = None):
        tipo, x, y = event[7:] # Do tipo choose_xxxx
        x = int(x); y = int(y)
        sel = self.intrinsic if tipo == "I" else self.extrinsic
        if update: sel[x,y] = value; self.update_all_buttons()
        return sel[x,y]

    def run_window(self):
        self.window.UnHide()

        while True:
            event, values = self.window.read(1)
            
            # Cases
            if event == sg.TIMEOUT_EVENT: pass
            elif event == sg.WINDOW_CLOSED: break
            elif event == "slider":       # Evento do slider
                self.window['input_text'].update(values[event])
                if self.selected: self.value_from_matrix(self.selected, True, values[event])
                self.previous_num = values[event]
            elif event.startswith("choose_"):   # Selecionar o valor das matrizes
                if self.selected: self.window[self.selected].update(disabled = False)
                self.selected = event
                value = self.value_from_matrix(event)
                self.window["slider"].update(value)
                self.window["input_text"].update(value)
                self.window[event].update(disabled = True)
            elif event.startswith("btn_"):      # Mudar o valor do slider por botão
                text = self.window[event].ButtonText
                new_value = 0.001 if text == "zero" else float(values["input_text"]) + float(text)
                self.window["slider"].update(new_value)
                self.window["input_text"].update(new_value)
                if self.selected: self.value_from_matrix(self.selected, True, new_value)
                self.previous_num = new_value
            elif event.startswith("sel_"):
                if event.endswith("fx"): self.window["choose_I00"].click()
                if event.endswith("fy"): self.window["choose_I11"].click()
                if event.endswith("cx"): self.window["choose_I02"].click()
                if event.endswith("cy"): self.window["choose_I12"].click()
            elif event == "input_ok":
                try: 
                    value = float(values["input_text"])
                    self.window["slider"].update(float(value))
                    if self.selected: self.value_from_matrix(self.selected, True, value); 
                    self.previous_num = value
                except: self.window["input_text"].update(self.previous_num)
            else: print(event, values)
            # print(values)


        self.window.close()

if __name__ == "__main__":
    slider = windows_control()
    slider.run_window()