import json

class ConfigDecoder:
    def __init__(self,path):
        self.data=""
        with open(path) as f:
            self.data = json.load(f)

    def get_model(self, name):
        json = self.data["archtecture"]["network"]
        return json[name]

    def get_const(self, name):
        json = self.data["archtecture"]["net_constant"]
        return json[name]

    def get_nms(self, name):
        json = self.data["archtecture"]["nms_setting"]
        return json[name]

    def get_training(self, name):
        json = self.data["training_setting"]
        return json[name]

    def get_option(self, name):
        json = self.data["option"]
        return json[name]

    def get_finetraining(self, name):
        json = self.data["finetunning_setting"]
        return json[name]

    def get_path(self, name):
        json = self.data["path"]
        return json[name]



if __name__ == "__main__":
    config = ConfigDecoder("./configure.json")
    ss = config.get_model("name")
    print(ss)
