import FreeSimpleGUI as sg
import numpy as np
from tkinter import Widget
import json

def prnt_num(num):
    return f"{num:+07.01f}"

def newbtn(name, key, disabled = False, font =("Courier", 12)):
    color = ("white", "black") if disabled else sg.theme_button_color()
    return sg.Button(name, key=key, font=font, disabled= disabled, button_color=color)
     

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
        self.distortion = np.array([[0,0,0,0,0]], dtype = np.float64)

        self.selected = None
        self.font = ("Courier", 10)

        self.layout = [
            [sg.Push(), self.create_slider_layout(), sg.Push()],
            [self.create_intrinsic_matrix(), sg.Push(),self.create_distortion_layout() ,sg.Push(), self.create_extrinsic_matrix()],
            [self.create_cases_selector(), sg.Push() , self.create_json_controls() ,sg.Push(), self.create_control_buttons()]
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
        for x in range(5): self.window[f"choose_D0{x}"].update(prnt_num(self.distortion[0][x]))

    def create_json_controls(self):
        return sg.Frame("Json Controls", [
            [newbtn("SAVE", key="json_save"), newbtn("LOAD", key="json_load")]
        ], title_location=sg.TITLE_LOCATION_TOP)

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
            [   sg.Push(),
                newbtn("-10e3", key="btn_2"), newbtn("-10e2", key="btn_3"),  newbtn("-10", key="btn_4"), newbtn("-1", key="btn_1"), newbtn("-0.1", key="btn_10"),
                newbtn("+0.1", key="btn_20"), newbtn("+1", key="btn_8"), newbtn("+10", key="btn_5"),newbtn("+10e2", key="btn_6"), newbtn("+10e3", key="btn_7"),
                sg.Push()
             ],
             [sg.Slider((-1080, 1080), orientation='h', expand_x=True, key="slider", default_value=0,
                        enable_events=True, disable_number_display=True, size=(75,20), resolution=30.0,), 
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

    def create_distortion_layout(self):
        return sg.Frame("Distortion Coeficients",[
            [newbtn("NONE", "choose_D00"),newbtn("NONE", "choose_D01")],[newbtn("NONE", "choose_D02"), newbtn("NONE", "choose_D03")], 
            [sg.Push(), newbtn("NONE", "choose_D04"), sg.Push()]
        ], title_location=sg.TITLE_LOCATION_TOP)

    def value_from_matrix(self, event, update = False, value = None):
        tipo, x, y = event[7:] # Do tipo choose_xxxx
        x = int(x); y = int(y)
        match tipo:
            case "I": sel = self.intrinsic
            case "E": sel = self.extrinsic
            case "D": sel = self.distortion
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
                if values[event] == 0: 
                    self.window["slider"].update(0.001)
                    self.window['input_text'].update(0.001)
                else:
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
            elif event.startswith("json_"):
                if event.endswith("load"): self.get_coefficients_from_json()
                elif event.endswith("save"): self.save_coefficientes_in_json()
            else: print(event, values)
            # print(values)


        self.window.close()

    def get_coefficients_from_json(self, file = "coefficients.json"):
        with open(file, "r") as f:
            a = json.load(f)
            self.intrinsic = np.array(a["Intrinsic"], np.float64)
            self.extrinsic = np.array(a["Extrinsic"], np.float64)
            self.distortion = np.array(a["Distortion"], np.float64)
        self.update_all_buttons()
        if self.selected:
            self.window["slider"].update(self.value_from_matrix(self.selected))
            self.window["input_text"].update(self.value_from_matrix(self.selected))

    def save_coefficientes_in_json(self, file = "new_coefficients.json"):
        f =  open(file, "w")
        f.write("{\n")
        f.write("\t\"Intrinsic\": ")
        f.write(f"{self.intrinsic.tolist()},\n")
        f.write("\t\"Distortion\": ")
        f.write(f"{self.distortion.tolist()},\n")
        f.write("\t\"Extrinsic\": ")
        f.write(f"{self.extrinsic.tolist()}\n")
        f.write("}")
        f.close()


if __name__ == "__main__":
    slider = windows_control()
    slider.run_window()