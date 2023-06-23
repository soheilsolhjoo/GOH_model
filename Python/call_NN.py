# invs          = [dwdai1, dwdai4_1, dwdai4_2]          : to be provided by subroutine UELASTOMER_ANISO
# model(invs)   = [energy, dwdai1, dwdai4(1), dwdai4(2)]: Required Output
# NOTE: other required output must be set to zero, that is, dwdai2 = 0, dwdai5 = 0

# Import packages
import joblib
# from tensorflow.keras.models import load_model
from tensorflow.keras.saving import load_model

def main(invs):
    # load the trained NN model
    model = load_model('GOH_NN.h5', compile=False)
    # load the scaler
    scaler = joblib.load('GOH_NN.scaler')
    # scale the invariants
    invs    = scaler.transform(invs)
    # [energy, dwdai1, dwdai4(1), dwdai4(2)]  = model(invs)
    soln = model(invs)
    energy      = soln[:,0].numpy()
    dwdai1      = soln[:,1].numpy()
    dwdai4_1    = soln[:,2].numpy()
    dwdai4_2    = soln[:,3].numpy()
    return energy, dwdai1, dwdai4_1, dwdai4_2

if __name__ == "__main__":
    # invs = [dwdai1, dwdai4_1, dwdai4_2]
    dwdai2 = 0
    dwdai5 = 0
    [energy, dwdai1, dwdai4_1, dwdai4_2] = main(invs)