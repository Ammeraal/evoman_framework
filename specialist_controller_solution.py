from evoman.controller import Controller

class player_controller(Controller):
    def control(self, inputs, controller):
        #inputs = controller.augment_inputs(raw_inputs)
        outputs = controller.activate(inputs)
        #controller_inputs = controller.process_outputs(outputs)
        return outputs