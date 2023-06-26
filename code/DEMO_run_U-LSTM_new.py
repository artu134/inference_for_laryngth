#!/bin/python
from data_provider import *
from lstm_SA_type_full_stable import *
import sys


# --- check if all input arguments are available
if __name__ == '__main__':
    
    # --- define some varibles
    xsz=(256, 256)      # xsize
    ysz=(256, 256)      # ysize
    ndim=3              # ndimension                
    nfil=64             # nfilters
    rnfil=None          # reduced_nfilters
    layer=5             # nlayers
    cl=4                # nclasses
    loss="softmax"      # loss_function
    opt="adam"          # optimizer
    clw=None            # class_weights 
    lrate=0.0001        # learning_rate
    drate=None          # decay_rate
    bnorm=True          # bn
    reg=None            # reg
    regsc=0.00001       # reg_scale
    imgstd=False        # image_std                
    cc=True             # crop_concat
    cnfilt=False        # constant_nfilters
    nm="LSTMfull_rgb_nat"  # name                      
    verb=False          # verbose
    md=30000            # model
    imgsuf="_rgb.png"   # image_suffix              
    labsuf="_mask.png"  # label_suffix
    bg=True             # background
    bs=10               # batch_size
    ep=40               # epochs
    seq_len=10          # sequence_length   
    run_mode=0      # 1: train, 0:test

    # define paths to the dataset
    PATH_tain="../dataset/train/*.png"
    PATH_valid="../dataset/val/*.png"
    PATH_test="../dataset/test/*.png"

    # If the path to an existing model is provided, the NN will be initialized with these weights, i.e. to continue training.
    LOAD_MODEL=None     # default: None  



    # PLEASE DO NOT CHANGE!
    if clw == None:  # class_weights          
        clw_str = "None"
    else:
        clw = float(clw)
        clw_str = str(round(clw, 1))

    if drate == None:  # decay_rate
        drate_str = "None"
    else:
        drate = float(drate)
        drate_str = str(round(drate, 1))

    if reg == None:  # reg
        regu_str = "None"
    else:
        regu_str = regu
    
    if bnorm == True:  # bnorm          
        bnorm_str = "True"
    elif bnorm == False:
        bnorm_str = "False"

    if bg == True:  # background          
        bg_str = "True"
    elif bg == False:
        bg_str = "False"


    # DataProvider doesn't count background as class
    ncl_data=int(cl)-1                

    # get model info
    MODEL_INFO = str("./LSTMfull_L" + str(layer) + "_F" + str(nfil) + "_LF" + loss + "_O" + opt + "_LR1.00E-04_C" + str(cl) + "_CW" + clw_str + "_DeR" + drate_str + "_BN" + bnorm_str + "_Reg" + regu_str + "_RegS1.00E-04_Std" + str(imgstd)  + "_CC" + str(cc) + "_constF" + str(cnfilt) + "_" + nm + "_DR" + drate_str + "_BS" + str(bs) + "_E" + str(ep) + "/")
    
    # define paths
    MODEL_PATH = str("./Training_Models/" + nm + "/") 
    LOG_PATH = str("./Training_Logs/" + nm + "/")
    
    # write vars to file
    f=open("log_NNconfig.txt","w+")
    f.write("unet_dyn-configuration:"+'\n')
    f.write("-----"+'\n')
    f.write("MODEL_INFO"+MODEL_INFO+'\n')
    f.write("-----"+'\n')
    f.write("xsize: "+str(xsz)+'\n')
    f.write("ysize: "+str(ysz)+'\n')
    f.write("ndimension: "+str(ndim)+'\n')
    f.write("nfilters: "+str(nfil)+'\n')
    f.write("reduced_nfilters: "+str(rnfil)+'\n')
    f.write("nlayers: "+str(layer)+'\n')
    f.write("nclasses: "+str(cl)+'\n')
    f.write("loss_function: "+str(loss)+'\n')
    f.write("optimizer: "+str(opt)+'\n')
    f.write("class_weights: "+str(clw)+'\n')
    f.write("learning_rate: "+str(lrate)+'\n')
    f.write("decay_rate: "+str(drate)+'\n')
    f.write("bn: "+str(bnorm)+'\n')
    f.write("reg: "+str(reg)+'\n')
    f.write("reg_scale: "+str(regsc)+'\n')
    f.write("image_std: "+str(imgstd)+'\n')
    f.write("crop_concat: "+str(cc)+'\n')
    f.write("constant_nfilters: "+str(cnfilt)+'\n')
    f.write("name: "+str(nm)+'\n')
    f.write("verbose: "+str(verb)+'\n')
    f.write("model: "+str(md)+'\n')
    f.write("image_suffix: "+str(imgsuf)+'\n')
    f.write("label_suffix: "+str(labsuf)+'\n')
    f.write("background: "+str(bg)+'\n')
    f.write("batch_size: "+str(bs)+'\n')
    f.write("epochs: "+str(ep)+'\n')
    f.write("run_mode(1:train, 0:test): "+str(run_mode)+'\n')
    f.write("-----"+'\n')


    if int(run_mode) == 1: #training
                    
        # load data
        data_train = SequenceDataProvider(PATH_tain, image_suffix=str(imgsuf), label_suffix=str(labsuf), nclasses=int(ncl_data), sequence_length=int(seq_len), background=bg)
        data_valid = SequenceDataProvider(PATH_valid, image_suffix=str(imgsuf), label_suffix=str(labsuf), nclasses=int(ncl_data), sequence_length=int(seq_len), background=bg)

        # define NN-architecture
        lstmnet = LSTM_SA_Architecture(nfilters=int(nfil), nlayers=int(layer), ndimensions=int(ndim), nclasses=int(cl), class_weights=clw, loss_function=str(loss), learning_rate=float(lrate), 
                                       nsequences=int(seq_len), decay_rate=None, bn=bnorm, reg=None, reg_scale=float(regsc), image_std=str(imgstd), constant_nfilters=str(cnfilt), name=str(nm))
        
        # train NN    
        trainer = LSTM_SA_Trainer(data_train, data_valid, batch_size=1, epochs=int(ep), display_step=1, log_path=LOG_PATH, model_path=MODEL_PATH, skip_val=False, load_model_path=LOAD_MODEL)
        trainer.train(lstmnet)
        
        
    elif int(run_mode) == 0: # testing
        
        # define paths
        MODEL_PATH = str("./Training_Models/LSTMfull_rgb_nat/model_to_run/model-30000")  #Memo: DR0.6 ist die default droprate
        # OUTPUT_PATH = str("./predictions/" + str(nm) + "/" + str(nm) + "_unet_" + str(layer) + "L_" + str(cl) + "C_" + str(loss)
        #  + "_imgstd" + str(imgstd) + "_CLW" + str(clw_str) + "/model" + str(md) + "/")
        # MODEL_PATH = "./lstm_save/"
        OUTPUT_PATH = "./predictions/"
        # load data
        data_test = SequenceDataProvider(PATH_test, image_suffix=str(imgsuf), label_suffix=None, nclasses=int(ncl_data), sequence_length=int(seq_len), background=bg, shuffle_data=False)
             

        # define NN-architecture          
        lstmnet = LSTM_SA_Architecture(nfilters=int(nfil), nlayers=int(layer), ndimensions=int(ndim), nclasses=int(cl), class_weights=clw, loss_function=str(loss), learning_rate=float(lrate), 
                                       nsequences=int(seq_len), decay_rate=None, bn=bnorm, reg=None, reg_scale=float(regsc), image_std=str(imgstd), constant_nfilters=str(cnfilt), name=str(nm))
             
        # run tester to generate predictions
        tester = LSTM_SA_Tester(data_test, lstmnet, MODEL_PATH, OUTPUT_PATH)
        tester.test(save_validate_image=True)
                

    else:
        # Something's wrong. Check run_mode. 
        f.write("unknown run_mode")
          
    
else:
    # Seems that either not all arguments were defined properly ore some arguments are missing. 
    f.write('Invalid Numbers of Arguments. Script will be terminated.')
    numarg = int(sys.argv)
    f.write('Number of Arguments: ' + str(numarg))
    

f.close()