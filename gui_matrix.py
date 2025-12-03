import FreeSimpleGUI as sg
import numpy as np
from tkinter import Widget

def prnt_num(num):
    return f"{num:+08.01f}"

def newbtn(name, key, disabled = False, font =("Courier", 12)):
    return sg.Button(name, key=key, font=font, disabled= disabled)
     

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
        self.font = ("Courier", 10)

        self.layout = [
            [sg.Push(), self.create_slider_layout(), sg.Push()],
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
            [newbtn("NONE", "choose_I00"), newbtn("NONE", "choose_I01", disabled=True), newbtn("NONE", "choose_I02")],
            [newbtn("NONE", "choose_I10", disabled=True), newbtn("NONE", "choose_I11"), newbtn("NONE", "choose_I12")],
            [newbtn("NONE", "choose_I20", disabled=True), newbtn("NONE", "choose_I21", disabled=True), newbtn("NONE", "choose_I22")]
        ], title_location=sg.TITLE_LOCATION_TOP, key="INTRINSIC")

    def create_extrinsic_matrix(self):
        return sg.Frame("Extrinsic Matrix",[
            [newbtn("NONE", "choose_E00"),newbtn("NONE", "choose_E01"),newbtn("NONE", "choose_E02")],
            [newbtn("NONE", "choose_E10"),newbtn("NONE", "choose_E11"),newbtn("NONE", "choose_E12")],
            [newbtn("NONE", "choose_E20"),newbtn("NONE", "choose_E21"),newbtn("NONE", "choose_E22")]
        ], title_location=sg.TITLE_LOCATION_TOP, key="EXTRINSIC")

    def create_slider_layout(self):
        return sg.Frame("Value Adjust", [
            [ #sg.Button("-10e4", key="btn_1"), 
                sg.Push(), newbtn("-10e3", key="btn_2"), newbtn("-10e2", key="btn_3"),  newbtn("-10", key="btn_4"), 
             newbtn("+10", key="btn_5"),newbtn("+10e2", key="btn_6"), newbtn("+10e3", key="btn_7"),  sg.Push()
             #sg.Button("+10e4", key="btn_8")
             ],
             [sg.Slider((-10e4 + 1, 10e4 - 1), (0), (0.01), orientation='h', expand_x=True, key="slider", 
                        enable_events=True, disable_number_display=True, size=(75,20)), 
              sg.Input("0.0", key="input_text", size=(8,20), font = ('Helvetica', 12)), sg.Ok(key="input_ok")
              ]
        ], title_location=sg.TITLE_LOCATION_BOTTOM, key="SLIDER", expand_x=True)

    def create_cases_selector(self):
        return sg.Frame("Instrinsic specifics", [
            [newbtn("Fx", key="sel_fx"), newbtn("Fy", key="sel_fy"), newbtn("Cx", key="sel_cx"), newbtn("Cy", key="sel_cy")]
        ], title_location= sg.TITLE_LOCATION_TOP)

    def create_control_buttons(self):
        return sg.Frame("Control Player", [
            [newbtn("<<", key="con_prev", font=('Helvetica', 12)), 
             newbtn("||", key="con_pause", font=('Helvetica', 12)), 
             newbtn(">", key="con_play", font=('Helvetica', 12)), 
             newbtn(">>", key="con_next", font=('Helvetica', 12))]
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