import pygame
import pygame_gui
import numpy as np
import pandas as pd
import imageio
import pyglet
from pygame_gui.elements import UITextEntryLine
from pygame_gui.elements import UIButton
from pygame_gui.elements import UILabel
from pygame_gui.windows import UIFileDialog
from pygame_gui.core.utility import create_resource_path
import SpectralClustering
import txt
import sys
import os
import unicodedata
from sklearn import datasets


df = None
file_dialog = None
isTxt = False
bg_Color = "#000000"
cur_path = os.path.abspath('.')
jpg_KMean = None
jpg_Spectral = None


pygame.init()

pygame.display.set_caption('KMean and Spectral Clustering')
window_surface = pygame.display.set_mode((800, 600))

background = pygame.Surface((800, 600))
#background.fill(pygame.Color('#000000'))

manager = pygame_gui.UIManager((800, 600), 'data/themes/theme_1.json')

title_label = UILabel(relative_rect=pygame.Rect((150, 25), (500, 50)),
                                    text = 'KMean and Spectral Clustering',
                                    manager = manager,
                                    object_id='#2')

importExcel_button = UIButton(relative_rect=pygame.Rect((150, 80), (200, 50)),
                                            text='Import File',
                                            manager=manager)

tryModel_button = UIButton(relative_rect=pygame.Rect((450, 80), (200, 50)),
                                            text='Try Model',
                                            manager=manager)

clusternum_label = UILabel(relative_rect=pygame.Rect((0, 135), (180, 50)),
                                    text = 'Enter Num of Clustering',                                     
                                    manager = manager,
                                    object_id='#1')


clusternum_entryline = UITextEntryLine(relative_rect=pygame.Rect((40, 190), (100, 50)),
                                    manager = manager)

knn_label = UILabel(relative_rect=pygame.Rect((180, 135), (180, 50)),
                                    text = 'Enter k Value of Knn',
                                    manager = manager,
                                    object_id='#1')

knn_entryline = UITextEntryLine(relative_rect=pygame.Rect((220, 190), (100, 50)),
                                    manager = manager)

sigma_label = UILabel(relative_rect=pygame.Rect((360, 135), (180, 50)),
                                    text = 'Enter sigma Value of Knn',
                                    manager = manager,
                                    object_id='#1')

sigma_entryline = UITextEntryLine(relative_rect=pygame.Rect((400, 190), (100, 50)),
                                    manager = manager)

filename_label = UILabel(relative_rect=pygame.Rect((540, 135), (250, 50)),
                                    text = 'Enter Stored image and gif name',
                                    manager = manager,
                                    object_id='#1')

filename_entryline = UITextEntryLine(relative_rect=pygame.Rect((565, 190), (200, 50)),
                                    manager = manager)

start_button = UIButton(relative_rect=pygame.Rect((300, 225), (200, 50)),
                                            text='Start',
                                            manager=manager)


warn_label = UILabel(relative_rect=pygame.Rect((0,275), (800, 25)),
                                    text = '',                                    
                                    manager = manager,
                                    #visible=0,
                                    object_id='#3')

graph_kmean_label = UILabel(relative_rect=pygame.Rect((100, 280), (200, 25)),
                                    text = 'KMean Clustering',
                                    manager = manager,
                                    object_id='#1')

graph_spectral_label = UILabel(relative_rect=pygame.Rect((510, 280), (200, 25)),
                                    text = 'Spectral Clustering',
                                    manager = manager,
                                    object_id='#1')

def is_number(i):
    try:
        float(i)
        return True
    except ValueError:
        pass
    return False

def reset_graph():
    global jpg_KMean
    global jpg_Spectral
    blank = pygame.Surface((0, 0))
    if jpg_KMean != None:
        background.blit(blank, (0, 305))
        jpg_KMean = None
    if jpg_Spectral != None:
        background.blit(blank, (380, 305))
        jpg_Spectral = None


def Start():
    name = filename_entryline.get_text()
    clusternum = clusternum_entryline.get_text()
    if clusternum.isdecimal():
        if df is not None:
            show_warntxt("Start Clustering...")
            clusternum = int(clusternum)
            success1 = startKmean(name + "KMean", clusternum)
            success2 = startClustering(name + "SpectralClustering", clusternum)
            if success1 and success2:
                show_warntxt("Done!")
            
        else:
            show_warntxt("please import the legal data")
    else:
         show_warntxt("please enter a integer")

def startClustering(name, clusternum):
    k_knn = knn_entryline.get_text()
    
    sigma = sigma_entryline.get_text()
    
    if k_knn.isdecimal():
        k_knn = int(k_knn)
        if k_knn == 0:
            k_knn = clusternum
            knn_entryline.set_text(str(k_knn))
    else:
        show_warntxt("please enter a legal number")
        return
    
    if is_number(sigma) :
        sigma = float(sigma)
        if sigma == 0:
            sigma = 0.1
            sigma_entryline.set_text(str(sigma))
    else:
        #print("please enter a legal number")
        show_warntxt("please enter a legal number")
        return

    if not isTxt:
        SpectralClustering.startClustering(clusternum, df, name, k_knn, sigma)
    else:
        txt.cluster_txt_data(df, clusternum, name, True, k_knn, sigma)
    #gif = imageio.get_reader('output\\'+ name+ '.gif', mode='I')
    '''
    pyglet.resource.path = '../output'
    animation = pyglet.resource.animation(name+ '.gif')
    sprite = pyglet.sprite.Sprite(animation, x=380, y=305)
    sprite.draw()
    '''
    global jpg_Spectral
    jpg_Spectral = pygame.image.load('output\\'+ name+ '.jpg')
    background.blit(jpg_Spectral, ((380, 305)))          
    return True
    
   
      
        
def startKmean(name, clusternum):
    if not isTxt:
        SpectralClustering.startKmean(clusternum, df, name)
    else:
        txt.cluster_txt_data(df, clusternum, name)
    global jpg_KMean
    jpg_KMean = pygame.image.load('output\\'+ name+ '.jpg')
    background.blit(jpg_KMean, ((0, 305)))      
    return True

       

def show_warntxt(text):
    #warn_label.visible = 1
    #warn_label.bg_colour.a = 0
    warn_label.set_text(text) 

def reset_warntxt():
    warn_label.set_text("") 

def show_fileDialog():
    global file_dialog
    file_dialog = UIFileDialog(pygame.Rect(160, 50, 440, 500),
                                                    manager,
                                                    window_title='Load File...',
                                                    initial_file_path='input/',
                                                    allow_existing_files_only=True,
                                                    visible = 1)
    file_dialog.background_colour.a = 1
    importExcel_button.disable()
    
    

def get_Excel(path):
    global df  
    try:       
        df = pd.read_excel(path)
        df = np.array(df)   
        show_warntxt("loading excel file successfully!")
    except pygame.error:
        show_warntxt("cannot load excel file!")
        pass

def get_Txt(path):
    global df
    try:
        df = pd.read_csv(path, sep = " ", header=None)
        df = np.array(df)     
        show_warntxt("loading txt file successfully!")
    except pygame.error:
        show_warntxt("cannot load txt file!")
        pass

def set_Modeldata():
    global df
    df, label = datasets.make_circles(500, factor=0.5, noise=0.05)
    
def set_Pre_Value(num = 2, k=5, sigma = 0.1, name = 'ModelData'):
    clusternum_entryline.set_text(str(num))
    knn_entryline.set_text(str(k))
    sigma_entryline.set_text(str(sigma))
    filename_entryline.set_text(name)
    
def set_Suggest_Value(path):
    print(path)
    if "R15" in path:
        set_Pre_Value(2, 5, 0.1, "R15_2")
    if "simple-map-dungeon" in path:
        set_Pre_Value(4, 3, 0.1, "simple_map_dungeon_4")
    if "map-dungeon2" in path:
        set_Pre_Value(4, 3, 0.1, "map-dungeon2_4")
    if "bridge3" in path:
        set_Pre_Value(10, 5, 0.1, "bridge3_10")
    if "pathbased" in path:
        set_Pre_Value(3, 5, 0.4, "pathbased_3")


clock = pygame.time.Clock()
is_running = True

while is_running:
    time_delta = clock.tick(60)/1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False
           
        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == importExcel_button:
                    show_fileDialog()
                elif event.ui_element == tryModel_button:
                    set_Modeldata()
                    set_Pre_Value()
                elif event.ui_element == start_button:
                    reset_warntxt()
                    reset_graph()
                    start_button.disable()
                    Start()
                    start_button.enable()
            elif event.user_type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
                
                try:
                    filepath = create_resource_path(event.text)
                    #print(filepath)
                    if filepath.find(".xlsx") >= 0 or filepath.find(".xls") >= 0 :
                        get_Excel(filepath)
                        isTxt = False
                        set_Suggest_Value(filepath)
                    elif filepath.find(".txt") >= 0:
                        get_Txt(filepath)
                        isTxt = True            
                        set_Suggest_Value(filepath)
                    else:
                        show_warntxt("please choose .xlsx or .xls or .txt file!")
                    
                except pygame.error:
                    pass
                
            elif event.user_type == pygame_gui.UI_WINDOW_CLOSE:
                if event.ui_element == file_dialog:                  
                    importExcel_button.enable()
                    file_dialog = None

        manager.process_events(event)

    manager.update(time_delta)

    window_surface.blit(background, (0, 0))
    manager.draw_ui(window_surface)

    pygame.display.update()

pygame.quit()
sys.exit()