import FreeSimpleGUI as sg
import re

def prnt_num(num):
    return f"{num:06.01f}"

class values_slider:
    def __init__(self):
        sg.theme("SystemDefaultForReal")

        self.intrinsic = [[1,0,0], [0,1,0], [0,0,1]]
        self.extrinsic = [[1,0,0], [0,1,0], [0,0,1]]
        self.selected = None

        self.layout = [
            [self.create_slider_layout()],
            [self.create_intrinsic_matrix(), sg.Push(), self.create_extrinsic_matrix()],
        ]
        self.window = sg.Window("SLIDER", self.layout, finalize=True)
        self.update_all_buttons()

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
            [sg.Button("-100", key="btn_1"), sg.Button("-10", key="btn_2"), sg.Button("-1", key="btn_3"), sg.Button("-0.1", key="btn_4"),
             sg.Button("+0.1", key="btn_5"), sg.Button("+1", key="btn_6"), sg.Button("+10", key="btn_7"), sg.Button("+100", key="btn_8")],
             [sg.Slider((-1000, 1000), (0), (0.01), orientation='h', expand_x=True, key="SLIDER_VALUE", enable_events=True)]
        ], title_location=sg.TITLE_LOCATION_BOTTOM, key="SLIDER")

    def value_from_matrix(self, event, update = False, value = None):
        tipo, x, y = event[7:] # Do tipo choose_xxxx
        x = int(x); y = int(y)
        sel = self.intrinsic if tipo == "I" else self.extrinsic
        if update: sel[x][y] = value
        else: return sel[x][y]

    def run_window(self):
        while True:
            event, values = self.window.read(1)
            match event:
                case sg.TIMEOUT_EVENT: pass
                case sg.WINDOW_CLOSED: break
                case "SLIDER_VALUE":
                    if self.selected: 
                        self.value_from_matrix(self.selected, True, values[event])
                        self.update_all_buttons()
                # Selecionar o botão e trocar a cor
                case s if re.match(r"^choose_", s):
                    if self.selected: print(self.window[self.selected].__dict__)
                    self.selected = event
                    self.window["SLIDER_VALUE"].update(self.value_from_matrix(event))
                # Trocar o valor do botão
                case s if re.match(r"^btn_", s): # Pressionar de um botão
                    new_value = min(max(-1000,values["SLIDER_VALUE"] + float(self.window[event].ButtonText)),1000)
                    self.window["SLIDER_VALUE"].update(new_value)
                    if self.selected: 
                        self.value_from_matrix(self.selected, True, new_value)
                        self.update_all_buttons()
                case _: print(event, values)
            # print(values)


        self.window.close()

if __name__ == "__main__":
    slider = values_slider()
    slider.run_window()