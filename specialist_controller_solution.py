from evoman.controller import Controller

class player_controller(Controller):
    def control(self, inputs, controller):
        outputs = controller.activate(inputs)
        #controller_inputs = controller.process(outputs)
        return outputs