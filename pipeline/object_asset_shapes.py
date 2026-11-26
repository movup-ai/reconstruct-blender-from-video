"""Additional recognizable primitive-based props for the background asset library."""
import math

GROUPS = {
 'Furniture': '''round_table side_table bedside_table console_table folding_table dining_table dining_chair folding_chair rocking_chair lounge_chair recliner beanbag bunk_bed crib cot dresser nightstand kitchen_cabinet display_cabinet filing_cabinet safe locker reception_desk workbench coat_rack clothes_rack hanger shoe_rack room_divider cushion pillow mattress rug doormat'''.split(),
 'Kitchen and household': '''kettle teapot saucepan frying_pan cooking_pot bowl plate tray bottle wine_bottle can jar drinking_glass knife fork spoon cutting_board toaster blender stand_mixer coffee_machine rice_cooker vacuum_cleaner upright_vacuum broom mop dustpan bucket drying_rack ironing_board iron sewing_machine toilet bidet urinal washbasin toilet_paper_roll towel towel_rail soap_dispenser toothbrush toothbrush_holder hair_dryer bathtub_faucet shower_head water_dispenser water_heater fan ceiling_fan air_conditioner tissue_box alarm_clock wall_clock grandfather_clock laundry_basket trash_bag umbrella umbrella_stand'''.split(),
 'Electronics and office': '''desktop_computer computer_tower laptop keyboard mouse telephone smartphone tablet speaker headphones game_console camera projector router printer scanner calculator book stack_of_books open_book newspaper folder notebook pen pencil scissors stapler tape_dispenser whiteboard noticeboard easel desk_organizer power_outlet light_switch thermostat security_camera smoke_detector fire_alarm electrical_panel cable_reel'''.split(),
 'Street and architecture': '''fire_hydrant mailbox parking_meter bollard road_sign stop_sign traffic_cone road_barrier concrete_barrier bike_rack manhole_cover storm_drain curb bus_shelter vending_machine atm ticket_machine turnstile gate garage_door roller_door doorway arch balcony handrail ramp escalator elevator drainpipe gutter vent chimney roof house shed flag flagpole antenna satellite_dish solar_panel fire_extinguisher fire_hose_reel electrical_box drinking_fountain outdoor_bin dumpster speed_bump construction_fence scaffold ladder power_line_pole'''.split(),
 'Transport and storage': '''bicycle motorcycle scooter kick_scooter wheelchair stroller shopping_cart wheelbarrow hand_truck trolley pallet_jack forklift tractor trailer pickup_truck ambulance fire_truck police_car boat canoe sailboat traffic_buoy shipping_container cardboard_box crate pallet toolbox suitcase backpack handbag briefcase basket shopping_basket shopping_bag duffel_bag storage_tub oil_drum gas_cylinder jerry_can fuel_pump'''.split(),
 'Park and nature': '''palm_tree deciduous_tree hedge cactus flower planter tree_stump log rock boulder fountain statue sculpture picnic_table gazebo park_bench swing slide seesaw sandbox tent beach_umbrella deck_chair campfire firewood pond rock_wall flower_bed'''.split(),
 'Sport and leisure': '''basketball_hoop soccer_goal tennis_net pingpong_table dumbbell barbell weight_bench treadmill exercise_bike punching_bag yoga_mat surfboard skateboard snowboard skis tennis_racket baseball_bat football basketball tennis_ball helmet guitar piano drum music_stand microphone microphone_stand toy_block teddy_bear ball chessboard'''.split(),
 'Workshop retail and medical': '''drill hammer handsaw shovel rake axe wrench pliers paint_bucket paint_roller paint_brush generator air_compressor industrial_pipe pipe_elbow work_light shopping_display checkout_counter cash_register shop_sign clothing_rack mannequin hospital_bed examination_table iv_stand stretcher medical_cart microscope first_aid_box safety_sign'''.split(),
 'Animals and food': '''dog cat horse bird cow sheep duck fish apple orange banana bread cake pizza egg'''.split(),
}
EXTRA_KINDS=[name for names in GROUPS.values() for name in names]
# These use source depth at their observed height instead of being extended to the floor.
GROUND = set('''round_table side_table bedside_table console_table folding_table dining_table dining_chair folding_chair rocking_chair lounge_chair recliner beanbag bunk_bed crib cot dresser nightstand kitchen_cabinet display_cabinet filing_cabinet safe locker reception_desk workbench coat_rack clothes_rack shoe_rack room_divider mattress rug doormat vacuum_cleaner upright_vacuum broom mop bucket drying_rack ironing_board toilet bidet washbasin water_dispenser water_heater fan grandfather_clock laundry_basket trash_bag umbrella_stand computer_tower easel fire_hydrant mailbox parking_meter bollard road_sign stop_sign traffic_cone road_barrier concrete_barrier bike_rack manhole_cover storm_drain curb bus_shelter vending_machine atm ticket_machine turnstile gate garage_door roller_door doorway arch balcony handrail ramp escalator elevator drainpipe drinking_fountain outdoor_bin dumpster speed_bump construction_fence scaffold ladder power_line_pole bicycle motorcycle scooter kick_scooter wheelchair stroller shopping_cart wheelbarrow hand_truck trolley pallet_jack forklift tractor trailer pickup_truck ambulance fire_truck police_car boat canoe sailboat traffic_buoy shipping_container cardboard_box crate pallet toolbox suitcase basket shopping_basket duffel_bag storage_tub oil_drum gas_cylinder jerry_can fuel_pump palm_tree deciduous_tree hedge cactus flower planter tree_stump log rock boulder fountain statue sculpture picnic_table gazebo park_bench swing slide seesaw sandbox tent beach_umbrella deck_chair campfire firewood pond rock_wall flower_bed basketball_hoop soccer_goal tennis_net pingpong_table weight_bench treadmill exercise_bike punching_bag yoga_mat piano drum music_stand microphone_stand generator air_compressor industrial_pipe shopping_display checkout_counter clothing_rack mannequin hospital_bed examination_table iv_stand stretcher medical_cart dog cat horse cow sheep'''.split())
FLOATING=set(EXTRA_KINDS)-GROUND
FLOATING -= {'work_light','paint_bucket','drum'}


def build_extra(kind,p):
    box,cyl,cone,beam,vessel=(p[k] for k in ('box','cyl','cone','beam','vessel'))
    table,chair,sofa,cabinet,shelf,frame,railing,car=(p[k] for k in ('table','chair','sofa','cabinet','shelf','frame','railing','car'))
    bpy=p['bpy'];material=p['material']
    def ellipsoid(name,at,size,subdivisions=2,dark=False):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdivisions,radius=1,location=at)
        ob=bpy.context.object;ob.name=name;ob.scale=tuple(v/2 for v in size);ob.data.materials.append(material(dark));return ob
    def ring(name,at,radius,tube=.025,axis='Z'):
        bpy.ops.mesh.primitive_torus_add(major_segments=16,minor_segments=4,major_radius=radius,minor_radius=tube,location=at,rotation={'Z':(0,0,0),'X':(0,math.pi/2,0),'Y':(math.pi/2,0,0)}[axis])
        ob=bpy.context.object;ob.name=name;ob.data.materials.append(material());return ob
    def wheels(width,depth,r=.15,z=None):
        for x in (-width/2,width/2):
            for y in (-depth/2,depth/2):cyl('Wheel',r,.08,(x,y,z or r),axis='X',dark=True)
    def screen(w=.5,h=.35,z=.5):box('Screen',(w,.035,h),(0,-.04,z),True)
    def board(w=1,h=1.5,stand=True):
        frame(w,h,.045)
        if stand:
            for x in (-w*.36,w*.36):beam('Leg',(x,0,0),(x,.3,-.65),.035)
    def animal(spec):
        w,l,h=spec
        ellipsoid('Body',(0,0,h*.62),(w,l,h*.5))
        ellipsoid('Head',(0,-l*.48,h*.87),(w*.65,l*.38,h*.34))
        for x in (-w*.3,w*.3):
            for y in (-l*.3,l*.3):beam('Leg',(x,y,h*.5),(x,y,.04),w*.16)
        for x in (-w*.20,w*.20):cone('Ear',w*.12,0,h*.18,(x,-l*.48,h*1.08))
        beam('Tail',(0,l*.44,h*.70),(0,l*.68,h*.82),w*.09)
    def bicycle(motor=False):
        for y in (-.65,.65):
            if motor:cyl('Tyre',.30,.17,(0,y,.30),axis='X',dark=True)
            else:
                ring('Tyre',(0,y,.32),.30,.035,axis='X')
                for a in range(0,180,45):
                    dy=.29*math.cos(math.radians(a));dz=.29*math.sin(math.radians(a))
                    beam('Spoke',(0,y-dy,.32-dz),(0,y+dy,.32+dz),.012)
        for a,b in [((0,-.65,.32),(0,-.34,.86)),((0,-.34,.86),(0,.2,.45)),((0,.2,.45),(0,.65,.32)),((0,.65,.32),(0,.15,.86)),((0,.15,.86),(0,.2,.45)),((0,.15,.86),(0,-.34,.86))]:beam('Frame',a,b,.055 if motor else .035)
        box('Seat',(.27,.35,.065),(0,.18,.92));beam('Handlebar',(-.3,-.37,1.0),(.3,-.37,1.0),.035)
        if motor:ellipsoid('Engine',(0,-.05,.56),(.38,.60,.38));box('Fuel tank',(.34,.45,.20),(0,-.10,.84))

    if kind in ('round_table','side_table','bedside_table','console_table','folding_table','dining_table'):
        if kind=='round_table':cyl('Top',.55,.06,(0,0,.73));cyl('Pedestal',.07,.7,(0,0,.35));cyl('Foot',.3,.04,(0,0,.02))
        else:
            w,d,h={'side_table':(.55,.50,.58),'bedside_table':(.48,.43,.55),'console_table':(1.3,.35,.82),'folding_table':(1.2,.6,.73),'dining_table':(1.7,.9,.76)}[kind];table(w,d,h)
    elif kind in ('dining_chair','folding_chair','rocking_chair','lounge_chair','deck_chair','recliner'):
        if kind in ('lounge_chair','deck_chair','recliner'):
            sofa(.85,1.1,.85);box('Footrest',(.65,.45,.12),(0,-.72,.32))
        else:chair()
        if kind=='rocking_chair':
            for x in (-.22,.22):beam('Runner',(x,-.40,.02),(x,.4,.02),.05)
        if kind=='folding_chair':
            for x in (-.22,.22):beam('Cross brace',(x,-.23,.03),(x,.23,.46),.025)
    elif kind=='beanbag':ellipsoid('Soft seat',(0,0,.3),(.95,.85,.6));ellipsoid('Back',(0,.25,.50),(.8,.40,.70))
    elif kind in ('bunk_bed','crib','cot','hospital_bed','stretcher','examination_table'):
        w= .65 if kind in ('stretcher','examination_table') else ( .75 if kind=='crib' else 1.)
        table(w,1.8,.6);box('Mattress',(w-.05,1.75,.16),(0,0,.71))
        if kind=='bunk_bed':
            box('Upper mattress',(w,1.8,.16),(0,0,1.65))
            for x in (-w/2,w/2):
                for y in (-.88,.88):box('Bedpost',(.055,.055,1.9),(x,y,.95))
            for z in (.4,.7,1.,1.3,1.6):beam('Ladder rung',(-w/2,-.95,z),(-w/2+.4,-.95,z),.035)
        if kind in ('crib','hospital_bed'):
            for y in (-.85,.85):
                for i in range(7):box('Rail slat',(.025,.04,.55),((i/6-.5)*w,y,.97))
                box('Top rail',(w,.05,.045),(0,y,1.25))
        if kind in ('hospital_bed','stretcher'):wheels(w-.12,1.4,.07)
    elif kind in ('dresser','nightstand','kitchen_cabinet','filing_cabinet','safe','locker'):
        w,h,d,n={'dresser':(1.2,1.,.5,4),'nightstand':(.5,.55,.42,2),'kitchen_cabinet':(.9,.8,.4,0),'filing_cabinet':(.45,1.3,.6,4),'safe':(.6,.7,.55,0),'locker':(.4,1.8,.45,0)}[kind]
        cabinet(w,h,d,n)
        if kind=='safe':cyl('Combination dial',.06,.045,(.1,-d/2-.065,.4),axis='Y',dark=True)
        if kind=='locker':
            for z in (1.45,1.52,1.59):box('Vent',(.22,.01,.012),(0,-d/2-.03,z),True)
    elif kind=='display_cabinet':
        shelf(1.0,1.7,.45,4)
        for x in (-.45,.45):box('Front post',(.04,.04,1.7),(x,-.22,.85))
    elif kind in ('reception_desk','checkout_counter'):cabinet(1.8,1.05,.7);box('Counter ledge',(1.9,.3,.08),(0,-.30,1.09))
    elif kind=='workbench':table(1.6,.65,.9);box('Lower shelf',(1.45,.53,.045),(0,0,.2))
    elif kind in ('coat_rack','clothes_rack','clothing_rack'):
        if kind=='coat_rack':
            cyl('Pole',.035,1.75,(0,0,.875));cyl('Base',.3,.04,(0,0,.02))
            for x,y in ((.25,0),(-.25,0),(0,.25),(0,-.25)):beam('Hook',(0,0,1.5),(x,y,1.7),.025)
        else:
            for x in (-.6,.6):
                beam('Upright',(x,0,0),(x,0,1.7),.04);beam('Foot',(x,-.3,.03),(x,.3,.03),.045)
            beam('Rail',(-.6,0,1.7),(.6,0,1.7),.04)
    elif kind=='hanger':
        for a,b in [((-.22,0,0),(0,0,.12)),((0,0,.12),(.22,0,0)),((-.22,0,0),(.22,0,0)),((0,0,.12),(0,0,.2))]:beam('Hanger',a,b,.015)
        ring('Hook',(0,0,.23),.035,.008,axis='Y')
    elif kind=='shoe_rack':shelf(.9,.6,.32,3)
    elif kind=='room_divider':
        for i in range(3):ob=box('Panel',(.58,.035,1.7),((i-1)*.55,0,.85));ob.rotation_euler.z=(-1)**i*.18
    elif kind in ('cushion','pillow','mattress','rug','doormat','yoga_mat','towel'):
        size={'cushion':(.5,.5,.14),'pillow':(.65,.4,.12),'mattress':(1.6,2,.22),'rug':(2,3,.015),'doormat':(.7,.45,.02),'yoga_mat':(.65,1.8,.012),'towel':(.5,.85,.01)}[kind];box('Pad',size,(0,0,size[2]/2))
    elif kind in ('kettle','teapot'):
        ellipsoid('Body',(0,0,.13),(.26,.23,.23));cyl('Lid',.075,.02,(0,0,.25));cyl('Knob',.02,.025,(0,0,.272))
        beam('Spout',(-.10,0,.14),(-.20,0,.22),.055);ring('Handle',(.14,0,.15),.08,.015,axis='Y')
    elif kind in ('saucepan','frying_pan','cooking_pot'):
        r=.14;h=.035 if kind=='frying_pan' else .13;vessel('Pan',h,r)
        if kind=='cooking_pot':
            for x in (-.18,.18):ring('Handle',(x,0,h*.7),.04,.012,axis='Y')
        else:beam('Handle',(.12,0,h*.7),(.38,0,h*.7),.035)
    elif kind in ('bowl','plate','tray'):
        if kind=='tray':
            box('Tray bottom',(.45,.3,.012),(0,0,.006))
            for x in (-.22,.22):box('Side',(.012,.3,.025),(x,0,.018))
            for y in (-.145,.145):box('Rim',(.45,.01,.025),(0,y,.018))
        else:vessel('Dish',.025 if kind=='plate' else .10,.14,profile=[(.07,0),(.14,.025 if kind=='plate' else .10),(.13,.025 if kind=='plate' else .10),(.06,.012)])
    elif kind in ('bottle','wine_bottle','soap_dispenser','product_bottle'):
        cyl('Body',.043,.18,(0,0,.09));cone('Shoulder',.043,.017,.045,(0,0,.2025));cyl('Neck',.017,.065,(0,0,.2575))
        if kind=='soap_dispenser':beam('Pump',(-.01,0,.30),(.055,0,.30),.018)
    elif kind in ('can','jar','drinking_glass','toothbrush_holder','tissue_box','toilet_paper_roll'):
        if kind=='tissue_box':box('Box',(.23,.12,.08),(0,0,.04));box('Tissue',(.12,.008,.08),(0,0,.12))
        elif kind=='toilet_paper_roll':ring('Paper roll',(0,0,.05),.04,.025)
        elif kind=='jar':cyl('Jar',.06,.14,(0,0,.07));cyl('Lid',.063,.02,(0,0,.15))
        elif kind=='can':cyl('Can',.033,.12,(0,0,.06))
        else:vessel('Cup',.13,.04)
    elif kind in ('knife','fork','spoon','toothbrush','pen','pencil'):
        beam('Handle',(0,0,0),(0,.14,0),.012 if kind in ('pen','pencil') else .018)
        if kind=='knife':box('Blade',(.035,.13,.004),(.008,.20,0))
        elif kind=='fork':
            for x in (-.012,0,.012):beam('Tine',(x,.14,0),(x,.205,0),.005)
        elif kind=='spoon':ellipsoid('Spoon bowl',(0,.17,0),(.038,.055,.012))
        elif kind=='toothbrush':box('Brush',(.014,.035,.012),(0,.16,.008))
        else:ob=cone('Tip',.006,0,.02,(0,.15,0));ob.rotation_euler.x=-math.pi/2
    elif kind=='cutting_board':box('Board',(.35,.25,.018),(0,0,.009));box('Grip',(.10,.08,.018),(0,.15,.009))
    elif kind=='toaster':box('Body',(.28,.17,.18),(0,0,.09));box('Slot',(.20,.025,.005),(0,-.035,.182),True);box('Slot',(.20,.025,.005),(0,.035,.182),True)
    elif kind in ('blender','stand_mixer','coffee_machine','rice_cooker','sewing_machine'):
        box('Base',(.30,.25,.07),(0,0,.035))
        if kind=='blender':cyl('Motor',.09,.15,(0,0,.13));ob=vessel('Jar',.22,.09);ob.location.z=.2;box('Lid',(.2,.2,.025),(0,0,.43))
        elif kind=='stand_mixer':
            box('Post',(.09,.13,.28),(.10,.04,.20));box('Head',(.3,.14,.10),(0,0,.36));vessel('Bowl',.15,.105)
        elif kind=='coffee_machine':box('Tower',(.26,.18,.36),(0,.04,.23));box('Spout',(.18,.18,.05),(0,-.06,.33));vessel('Cup',.07,.04)
        elif kind=='rice_cooker':ellipsoid('Pot',(0,0,.18),(.28,.28,.25));cyl('Lid',.14,.035,(0,0,.30))
        else:box('Post',(.08,.15,.24),(.10,0,.18));box('Arm',(.28,.12,.08),(0,0,.34));beam('Needle',(-.11,0,.3),(-.11,0,.1),.012)
    elif kind in ('vacuum_cleaner','upright_vacuum'):
        if kind=='vacuum_cleaner':ellipsoid('Body',(0,.25,.18),(.38,.45,.32));wheels(.35,.28,.08)
        else:box('Body',(.22,.20,.6),(0,.1,.45))
        box('Head',(.34,.2,.08),(0,-.2,.04));beam('Handle',(0,-.15,.06),(0,.10,1.1),.035)
    elif kind in ('broom','mop','shovel','rake','axe','paint_roller'):
        beam('Handle',(0,0,.10),(0,0,1.3),.025)
        if kind in ('broom','mop'):box('Head',(.32,.10,.13),(0,0,.08))
        elif kind=='shovel':box('Blade',(.24,.045,.32),(0,-.015,.16))
        elif kind=='rake':
            beam('Head',(-.2,0,.13),(.2,0,.13),.025)
            for i in range(9):beam('Tooth',((i-4)*.045,0,.13),((i-4)*.045,-.1,.02),.015)
        elif kind=='axe':box('Axe head',(.28,.05,.14),(.06,0,1.20))
        else:cyl('Roller',.045,.25,(0,0,1.3),axis='X')
    elif kind=='dustpan':box('Pan',(.25,.25,.018),(0,0,.009));beam('Grip',(0,.10,0),(0,.3,.08),.03)
    elif kind in ('bucket','paint_bucket','laundry_basket','basket','shopping_basket','planter'):
        if kind in ('basket','shopping_basket','laundry_basket'):
            box('Base',(.5,.35,.03),(0,0,.015))
            for x in (-.24,.24):box('Side',(.02,.35,.30),(x,0,.15))
            for y in (-.165,.165):box('Side',(.5,.02,.30),(0,y,.15))
            if kind!='laundry_basket':ring('Handle',(0,0,.35),.2,.02,axis='Y')
        else:vessel('Bucket',.38,.22)
    elif kind=='drying_rack':
        for x in (-.6,.6):beam('Leg',(x,-.3,0),(-x,.3,.9),.025)
        for i in range(8):beam('Rail',(-.6,(i/7-.5)*.6,.9),(.6,(i/7-.5)*.6,.9),.015)
    elif kind=='ironing_board':table(1.2,.38,.85)
    elif kind=='iron':box('Base',(.13,.24,.035),(0,0,.018));box('Body',(.1,.18,.09),(0,.02,.07));ring('Handle',(0,.02,.14),.055,.015,axis='X')
    elif kind in ('toilet','bidet','urinal','washbasin'):
        if kind=='urinal':
            ellipsoid('Bowl',(0,0,.4),(.40,.34,.60));box('Wall back',(.4,.06,.75),(0,.12,.43));ellipsoid('Inset',(0,-.14,.44),(.28,.025,.34),dark=True)
        else:
            cyl('Pedestal',.13,.32,(0,.06,.16));ellipsoid('Bowl',(0,-.06,.36),(.43,.58,.25));ring('Rim',(0,-.06,.46),.18,.035)
            if kind=='toilet':box('Tank',(.40,.16,.42),(0,.24,.55))
            elif kind=='washbasin':vessel('Basin',.08,.26);beam('Tap',(0,.18,.48),(0,.1,.58),.03)
    elif kind=='towel_rail':
        beam('Rail',(-.35,0,.05),(.35,0,.05),.02)
        for x in (-.35,.35):beam('Bracket',(x,0,.05),(x,.08,.05),.02)
    elif kind in ('hair_dryer','drill'):
        cyl('Body',.06,.22,(0,0,.17),axis='Y');box('Grip',(.07,.07,.16),(0,.05,.07))
        if kind=='drill':cyl('Bit',.008,.12,(0,-.17,.17),axis='Y')
    elif kind in ('bathtub_faucet','shower_head'):
        beam('Pipe',(0,0,0),(0,0,.3),.03);beam('Spout',(0,0,.3),(0,-.2,.3),.03)
        if kind=='shower_head':cyl('Head',.08,.025,(0,-.2,.28))
    elif kind in ('water_dispenser','water_heater'):
        if kind=='water_heater':cyl('Tank',.23,1.,(0,0,.5))
        else:box('Cabinet',(.35,.35,.9),(0,0,.45));cyl('Bottle',.15,.35,(0,0,1.075));box('Tap panel',(.25,.025,.2),(0,-.19,.65),True)
    elif kind in ('fan','ceiling_fan'):
        if kind=='fan':cyl('Stand',.025,.85,(0,0,.425));cyl('Foot',.22,.035,(0,0,.018));ring('Guard',(0,0,1.0),.25,.012,axis='Y');cyl('Hub',.06,.10,(0,0,1.0),axis='Y')
        else:cyl('Stem',.035,.35,(0,0,.175));cyl('Motor',.12,.08,(0,0,0))
        for i in range(3):
            a=i*math.tau/3
            if kind=='fan':beam('Blade',(0,-.03,1.),(.20*math.cos(a),-.03,1.+.20*math.sin(a)),.07,.015)
            else:beam('Blade',(0,0,0),(.65*math.cos(a),.65*math.sin(a),0),.12,.025)
    elif kind=='air_conditioner':box('Body',(.85,.22,.28),(0,0,.14));box('Outlet',(.74,.025,.05),(0,-.12,.055),True)
    elif kind in ('alarm_clock','wall_clock','grandfather_clock'):
        if kind=='grandfather_clock':cabinet(.5,1.9,.3);z=1.5;r=.20;y=-.19
        else:z=.13;r=.13;y=0;cyl('Clock face',r,.04,(0,0,z),axis='Y')
        if kind=='grandfather_clock':cyl('Clock face',r,.03,(0,y,z),axis='Y')
        beam('Hour hand',(0,y-.025,z),(.07,y-.025,z+.04),.008);beam('Minute hand',(0,y-.026,z),(0,y-.026,z+.10),.006)
    elif kind=='trash_bag':ellipsoid('Bag',(0,0,.30),(.45,.40,.60));cone('Tie',.07,0,.14,(0,0,.63))
    elif kind in ('umbrella','beach_umbrella'):
        h=2.1 if kind=='beach_umbrella' else .9;r=1 if kind=='beach_umbrella' else .5
        cyl('Pole',.018,h,(0,0,h/2));cone('Canopy',r,0,r*.35,(0,0,h))
    elif kind=='umbrella_stand':vessel('Stand',.5,.14)
    elif kind in ('desktop_computer','laptop','keyboard','mouse','smartphone','tablet','computer_tower'):
        if kind=='computer_tower':cabinet(.22,.48,.45)
        elif kind in ('smartphone','tablet'):
            box('Device',(.08,.012,.16) if kind=='smartphone' else (.20,.012,.28),(0,0,.08 if kind=='smartphone' else .14))
            box('Screen',(.065,.002,.14) if kind=='smartphone' else (.18,.002,.25),(0,-.007,.08 if kind=='smartphone' else .14),True)
        elif kind=='mouse':ellipsoid('Mouse',(0,0,.025),(.065,.11,.05));beam('Split',(0,-.04,.048),(0,.01,.048),.002)
        elif kind=='keyboard':
            box('Keyboard',(.43,.14,.018),(0,0,.009))
            for i in range(12):box('Key',(.025,.07,.006),((i-5.5)*.032,0,.021))
        elif kind=='laptop':box('Base',(.34,.24,.016),(0,-.10,.008));screen(.34,.22,.12)
        else:screen(.5,.32,.35);box('Foot',(.25,.15,.02),(0,0,.01));beam('Stand',(0,0,.02),(0,0,.3),.03)
    elif kind in ('telephone','calculator','game_console','router','printer','scanner','projector','cash_register'):
        sizes={'telephone':(.20,.18,.05),'calculator':(.09,.16,.018),'game_console':(.3,.24,.07),'router':(.2,.14,.035),'printer':(.45,.38,.23),'scanner':(.42,.28,.04),'projector':(.30,.23,.10),'cash_register':(.40,.35,.20)}
        w,d,h=sizes[kind];box('Body',(w,d,h),(0,0,h/2))
        if kind=='telephone':box('Handset',(.22,.045,.04),(0,.04,.075))
        elif kind=='router':
            for x in (-.08,.08):beam('Antenna',(x,.04,.03),(x,.04,.22),.01)
        elif kind=='projector':cyl('Lens',.035,.05,(.07,-d/2-.025,h/2),axis='Y',dark=True)
        elif kind=='printer':box('Paper feed',(.3,.08,.10),(0,.16,.26));box('Paper slot',(.3,.02,.015),(0,-.2,.10),True)
        elif kind=='cash_register':box('Display',(.20,.06,.14),(0,.07,.29));box('Keys',(.26,.14,.025),(0,-.06,.215),True)
        else:box('Panel',(w*.7,d*.6,.006),(0,0,h+.003),True)
    elif kind=='speaker':
        box('Cabinet',(.25,.23,.55),(0,0,.275))
        for z,r in ((.17,.085),(.40,.045)):cyl('Driver',r,.018,(0,-.125,z),axis='Y',dark=True)
    elif kind=='headphones':
        ring('Headband',(0,0,.13),.12,.012,axis='Y')
        for x in (-.115,.115):ellipsoid('Earcup',(x,0,.10),(.05,.08,.11))
    elif kind=='camera':box('Body',(.13,.05,.09),(0,0,.045));cyl('Lens',.035,.06,(0,-.05,.045),axis='Y',dark=True)
    elif kind in ('book','stack_of_books','open_book','notebook','folder','newspaper'):
        if kind=='stack_of_books':
            for i in range(4):ob=box('Book',(.22,.30,.035),(0,0,.018+i*.04));ob.rotation_euler.z=i*.08
        elif kind=='open_book':
            for x in (-.105,.105):ob=box('Pages',(.20,.28,.02),(x,0,.025));ob.rotation_euler.y=-.13 if x<0 else .13
        else:box('Pages',(.21,.29,.035 if kind=='book' else .008),(0,0,.018 if kind=='book' else .004))
    elif kind in ('scissors','pliers'):
        for x in (-.025,.025):ring('Grip',(x,-.06,0),.025,.008);beam('Blade',(x,-.04,0),(-x,.10,.005),.012)
    elif kind=='stapler':box('Base',(.035,.14,.012),(0,0,.006));box('Arm',(.035,.14,.025),(0,0,.04));box('Hinge',(.04,.025,.04),(0,.055,.025))
    elif kind=='tape_dispenser':ring('Tape',(0,0,.04),.035,.014,axis='X');box('Foot',(.07,.14,.02),(0,-.02,.01))
    elif kind in ('whiteboard','noticeboard','easel'):board(1.4 if kind!='easel' else .7,.85 if kind!='easel' else 1.2,kind=='easel')
    elif kind=='desk_organizer':
        for x in (-.06,.06):ob=vessel('Pen holder',.12,.045);ob.location.x=x
        box('Base',(.24,.13,.015),(0,0,.008))
    elif kind in ('power_outlet','light_switch','thermostat','fire_alarm','electrical_box','electrical_panel','first_aid_box','safety_sign','shop_sign'):
        w,h,d=(.6,.9,.16) if kind in ('electrical_box','electrical_panel') else ((.6,.4,.04) if kind in ('safety_sign','shop_sign') else (.12,.16,.025))
        box('Plate',(w,d,h),(0,0,h/2))
        if kind in ('power_outlet','light_switch'):box('Inset',(w*.5,.015,h*.45),(0,-d/2-.008,h/2),True)
        elif kind=='first_aid_box':box('Cross',(.025,.01,.09),(0,-d/2-.01,h/2),True);box('Cross',(.08,.01,.025),(0,-d/2-.012,h/2),True)
        else:box('Display',(w*.65,.01,h*.5),(0,-d/2-.006,h*.55),True)
    elif kind=='security_camera':box('Body',(.12,.23,.10),(0,0,.18));cyl('Lens',.03,.02,(0,-.13,.18),axis='Y',dark=True);beam('Mount',(0,.08,.15),(0,.12,.02),.035)
    elif kind=='smoke_detector':cyl('Housing',.07,.04,(0,0,.02));cyl('Face',.045,.01,(0,0,.045),dark=True)
    elif kind in ('cable_reel','fire_hose_reel'):
        cyl('Drum',.22,.16,(0,0,.30),axis='Y');
        for y in (-.10,.10):cyl('Flange',.25,.025,(0,y,.30),axis='Y')
        if kind=='cable_reel':box('Stand',(.38,.30,.04),(0,0,.02))
    elif kind=='fire_hydrant':
        cyl('Body',.11,.65,(0,0,.325));ellipsoid('Top',(0,0,.67),(.23,.23,.17));cyl('Cross pipe',.075,.42,(0,0,.45),axis='X');cyl('Base',.16,.055,(0,0,.028))
    elif kind in ('mailbox','parking_meter','bollard','road_sign','stop_sign','safety_sign'):
        h=1.6 if kind!='bollard' else .85;cyl('Post',.035 if kind!='bollard' else .085,h,(0,0,h/2))
        if kind=='mailbox':box('Box',(.4,.5,.32),(0,0,1.25));box('Slot',(.25,.01,.025),(0,-.26,1.27),True)
        elif kind=='parking_meter':box('Meter',(.2,.16,.35),(0,0,1.6));screen(.13,.15,1.65)
        elif kind=='stop_sign':cyl('Octagonal sign',.35,.035,(0,0,h),axis='Y',vertices=8)
        elif kind=='road_sign':box('Sign',(.8,.035,.45),(0,0,h))
    elif kind=='traffic_cone':box('Base',(.40,.40,.035),(0,0,.018));cone('Cone',.16,.025,.60,(0,0,.335))
    elif kind in ('road_barrier','concrete_barrier','construction_fence'):
        if kind=='construction_fence':railing(2.4,1.8)
        elif kind=='road_barrier':
            for x in (-.75,.75):box('Foot',(.10,.55,.07),(x,0,.035));box('Post',(.07,.07,.9),(x,0,.45))
            for z in (.55,.82):box('Board',(1.7,.06,.16),(0,0,z))
        else:box('Base',(2.,.5,.15),(0,0,.075));box('Body',(2.,.22,.7),(0,0,.5))
    elif kind=='bike_rack':
        for x in (-.7,-.35,0,.35,.7):
            for y in (-.18,.18):beam('Upright',(x,y,0),(x,y,.55),.025)
            beam('Top',(x,-.18,.55),(x,.18,.55),.025)
    elif kind in ('manhole_cover','storm_drain','speed_bump','curb'):
        if kind=='manhole_cover':cyl('Cover',.35,.025,(0,0,.013))
        elif kind=='storm_drain':
            for x in range(9):box('Grate bar',(.025,.40,.025),((x-4)*.045,0,.013))
            for y in (-.20,.20):box('Rim',(.42,.025,.025),(0,y,.013))
        else:box('Curb' if kind=='curb' else 'Bump',(1.5,.18 if kind=='curb' else .3,.15 if kind=='curb' else .08),(0,0,.075 if kind=='curb' else .04))
    elif kind in ('bus_shelter','shed','gazebo'):
        w,d,h=(2.4,1.5,2.4)
        for x in (-w/2,w/2):
            for y in (-d/2,d/2):box('Post',(.08,.08,h),(x,y,h/2))
        box('Roof',(w+.2,d+.2,.12),(0,0,h))
        if kind!='gazebo':box('Back',(w,.06,h),(0,d/2,h/2))
        if kind=='shed':box('Side',(.06,d,h),(-w/2,0,h/2));box('Side',(.06,d,h),(w/2,0,h/2))
    elif kind in ('vending_machine','atm','ticket_machine','fuel_pump'):
        cabinet(.85,1.9,.65);box('Screen',(.6,.02,.85),(0,-.37,1.2),True);box('Delivery slot',(.5,.04,.16),(0,-.4,.3),True)
        if kind=='fuel_pump':beam('Hose',(.45,0,1.4),(.60,0,.35),.045)
    elif kind=='turnstile':
        box('Pedestal',(.32,.4,1.),(0,0,.5))
        for a in range(3):beam('Arm',(.18,0,.9),(.6,math.cos(a*math.tau/3)*.35,.9+math.sin(a*math.tau/3)*.35),.035)
    elif kind in ('gate','garage_door','roller_door','elevator'):
        frame(2.4 if kind!='elevator' else 1.4,2.2,.10)
        if kind in ('garage_door','roller_door'):
            for z in range(12):box('Panel',(2.25,.06,.16),(0,-.05,.12+z*.18))
        elif kind=='gate':railing(2.2,2.1)
        else:box('Centre seam',(.015,.015,2.),(0,-.06,1.1),True)
    elif kind in ('doorway','arch'):
        for x in (-.65,.65):box('Upright',(.20,.22,2.),(x,0,1.))
        if kind=='doorway':box('Lintel',(1.5,.22,.2),(0,0,2.1))
        else:
            for i in range(9):
                a=math.pi*i/8;ob=box('Arch stone',(.25,.22,.20),(.65*math.cos(a),0,2.+.65*math.sin(a)));ob.rotation_euler.y=-a
    elif kind=='balcony':box('Slab',(2.4,1.0,.15),(0,0,.075));railing(2.4,1.1)
    elif kind=='handrail':
        beam('Rail',(-1,0,.9),(1,0,1.6),.045)
        for x,z in ((-1,.9),(0,1.25),(1,1.6)):beam('Post',(x,0,0),(x,0,z),.03)
    elif kind in ('ramp','escalator'):
        if kind=='ramp':
            verts=[(-.7,-1.5,0),(.7,-1.5,0),(-.7,1.5,0),(.7,1.5,0),(-.7,1.5,1),(.7,1.5,1)]
            mesh=bpy.data.meshes.new('Wedge');mesh.from_pydata(verts,[],[(0,2,3,1),(0,1,5,4),(0,4,2),(1,3,5),(2,4,5,3)]);mesh.update();ob=bpy.data.objects.new('Ramp',mesh);bpy.context.collection.objects.link(ob);ob.data.materials.append(material())
        else:
            for i in range(10):box('Step',(1.,.27,.16*(i+1)),(0,(i-4.5)*.27,.08*(i+1)))
            for x in (-.6,.6):beam('Handrail',(x,-1.4,.9),(x,1.4,2.5),.08)
    elif kind in ('drainpipe','gutter','industrial_pipe','pipe_elbow'):
        if kind=='gutter':vessel('Gutter',2.,.08)
        else:
            cyl('Pipe',.06 if kind=='drainpipe' else .16,2.,(0,0,1.))
            if kind=='pipe_elbow':cyl('Elbow',.16,.7,(0,-.3,1.85),axis='Y')
    elif kind=='vent':
        frame(.6,.4,.05)
        for i in range(8):box('Louver',(.5,.05,.025),(0,-.02,.04+i*.045))
    elif kind=='chimney':box('Stack',(.5,.5,1.3),(0,0,.65));box('Cap',(.65,.65,.12),(0,0,1.35))
    elif kind in ('roof','house'):
        if kind=='house':box('Walls',(3.,3.,2.4),(0,0,1.2));box('Door',(.7,.025,1.6),(0,-1.52,.8),True)
        z=2.4 if kind=='house' else 0
        for x in (-.8,.8):ob=box('Roof plane',(1.9,3.4,.10),(x,0,z+.50));ob.rotation_euler.y=math.radians(30 if x>0 else -30)
    elif kind in ('flag','flagpole'):
        cyl('Pole',.03,3.,(0,0,1.5));box('Flag',(.95,.015,.55),(.48,0,2.6))
    elif kind=='antenna':
        beam('Mast',(0,0,0),(0,0,1.4),.03)
        for z in (.7,1.,1.3):beam('Element',(-.6,0,z),(.6,0,z),.015)
    elif kind=='satellite_dish':
        ellipsoid('Dish',(0,0,.6),(.8,.10,.8));beam('Support',(0,.1,0),(0,.1,.6),.05);beam('Receiver arm',(0,-.05,.3),(0,-.5,.65),.035)
    elif kind=='solar_panel':
        ob=box('Panel',(1.,1.65,.04),(0,0,.4));ob.rotation_euler.x=math.radians(20)
        for x in (-.4,.4):beam('Support',(x,0,0),(x,0,.4),.03)
    elif kind=='fire_extinguisher':cyl('Tank',.09,.46,(0,0,.23));box('Handle',(.14,.05,.045),(0,0,.50));beam('Hose',(.06,0,.47),(.12,0,.13),.015)
    elif kind=='drinking_fountain':box('Pedestal',(.3,.3,.85),(0,0,.425));ob=vessel('Basin',.06,.23);ob.location.z=.9;box('Top',(.45,.40,.05),(0,0,.88))
    elif kind=='outdoor_bin':vessel('Bin',.9,.28);cyl('Lid',.30,.05,(0,0,.925))
    elif kind=='dumpster':
        box('Body',(1.8,1.0,1.05),(0,0,.65));box('Lid',(1.9,1.08,.08),(0,0,1.215));wheels(1.5,.8,.10)
    elif kind in ('scaffold','ladder','power_line_pole'):
        if kind=='power_line_pole':
            cyl('Pole',.10,5.,(0,0,2.5));beam('Crossarm',(-.8,0,4.5),(.8,0,4.5),.08)
            for x in (-.65,0,.65):cyl('Insulator',.07,.16,(x,0,4.62))
        elif kind=='ladder':
            for x in (-.22,.22):beam('Rail',(x,0,0),(x,0,2.1),.04)
            for i in range(8):beam('Rung',(-.22,0,.15+i*.25),(.22,0,.15+i*.25),.03)
        else:
            for x in (-.7,.7):
                for y in (-.35,.35):beam('Post',(x,y,0),(x,y,2.4),.035)
            for z in (.2,1.2,2.2):box('Platform',(1.4,.7,.045),(0,0,z))
            beam('Brace',(-.7,-.35,0),(.7,-.35,2.4),.03)
    elif kind in ('bicycle','motorcycle'):bicycle(kind=='motorcycle')
    elif kind in ('scooter','kick_scooter'):
        for y in (-.55,.55):cyl('Wheel',.13 if kind=='kick_scooter' else .22,.08,(0,y,.22),axis='X',dark=True)
        box('Deck',(.24,1.,.08),(0,0,.25));beam('Stem',(0,-.5,.25),(0,-.4,1.1),.04);beam('Handlebar',(-.25,-.4,1.1),(.25,-.4,1.1),.035)
        if kind=='scooter':box('Seat',(.3,.35,.07),(0,.24,.8));box('Front shield',(.38,.15,.6),(0,-.4,.6))
    elif kind in ('wheelchair','stroller'):
        for x in (-.30,.30):ring('Wheel',(x,.15,.30),.28,.025,axis='X')
        for x in (-.23,.23):cyl('Front wheel',.08,.05,(x,-.40,.08),axis='X')
        box('Seat',(.5,.5,.06),(0,-.05,.45));box('Back',(.5,.06,.55),(0,.2,.7))
        for x in (-.26,.26):beam('Armrest',(x,-.25,.65),(x,.25,.65),.035)
        if kind=='stroller':box('Canopy',(.6,.5,.08),(0,.05,1.0));beam('Pushbar',(-.25,.4,.95),(.25,.4,.95),.03)
    elif kind in ('shopping_cart','wheelbarrow','hand_truck','trolley','pallet_jack','medical_cart'):
        if kind=='hand_truck':
            box('Toe',(.45,.35,.04),(0,-.12,.05))
            for x in (-.2,.2):beam('Rail',(x,0,.05),(x,0,1.2),.035);cyl('Wheel',.13,.07,(x,.08,.13),axis='X')
            beam('Handle',(-.2,0,1.2),(.2,0,1.2),.035)
        elif kind=='wheelbarrow':
            box('Basin',(.6,.8,.25),(0,0,.55));cyl('Wheel',.20,.10,(0,-.5,.20),axis='X')
            for x in (-.23,.23):beam('Handle',(x,-.3,.4),(x,.9,.7),.035)
        elif kind=='pallet_jack':
            for x in (-.22,.22):box('Fork',(.14,1.2,.08),(x,0,.12))
            beam('Handle',(0,.5,.15),(0,.6,1.2),.05);wheels(.4,.9,.08)
        else:
            table(.65,.85,.85);wheels(.55,.75,.07)
            if kind=='shopping_cart':
                for x in (-.3,.3):box('Basket wall',(.035,.8,.40),(x,0,.95))
                for y in (-.38,.38):box('Basket wall',(.6,.035,.40),(0,y,.95))
            else:box('Lower tray',(.65,.85,.04),(0,0,.22))
    elif kind in ('forklift','tractor'):
        box('Body',(1.2,1.8,.65),(0,0,.70));wheels(1.1,1.3,.35)
        box('Seat',(.5,.5,.15),(0,.2,1.1))
        if kind=='forklift':
            for x in (-.35,.35):beam('Mast',(x,-.9,.1),(x,-.9,2.2),.1);box('Fork',(.12,1.2,.09),(x,-1.4,.15))
        else:box('Engine hood',(.75,.9,.45),(0,-.4,1.18));cyl('Exhaust',.035,.55,(.3,-.4,1.65))
    elif kind=='trailer':box('Bed',(1.8,3.,.15),(0,0,.65));wheels(1.8,1.5,.3);beam('Towbar',(0,-1.5,.6),(0,-2.4,.5),.12)
    elif kind in ('pickup_truck','ambulance','fire_truck','police_car'):
        if kind=='pickup_truck':
            box('Chassis',(1.9,4.8,.55),(0,0,.65));wheels(1.9,3.1,.32)
            box('Cabin',(1.7,1.8,.85),(0,-.7,1.35));box('Windshield',(1.4,.025,.5),(0,-1.62,1.5),True)
            for x in (-.9,.9):box('Bed side',(.10,1.9,.45),(x,1.2,1.1))
            box('Tailgate',(1.9,.10,.45),(0,2.2,1.1))
        else:car('truck' if kind in ('ambulance','fire_truck') else 'car')
        if kind in ('ambulance','fire_truck','police_car'):box('Lightbar',(.75,.20,.12),(0,-.4,2.6 if kind!='police_car' else 1.53),True)
        if kind=='fire_truck':
            for x in (-.4,.4):beam('Ladder rail',(x,-1.,2.7),(x,2.,2.7),.05)
            for y in range(8):beam('Rung',(-.4,-1+y*.4,2.7),(.4,-1+y*.4,2.7),.04)
    elif kind in ('boat','canoe','sailboat'):
        ellipsoid('Hull',(0,0,.25),(.9 if kind=='canoe' else 1.5,3.5,.55))
        box('Deck',(.7 if kind=='canoe' else 1.1,2.8,.08),(0,0,.48))
        if kind=='sailboat':beam('Mast',(0,0,.5),(0,0,3.3),.055);box('Sail',(.025,1.5,1.8),(0,.55,2.1))
    elif kind=='traffic_buoy':cone('Float',.35,.18,.55,(0,0,.275));cyl('Marker',.08,.8,(0,0,.8))
    elif kind=='shipping_container':
        box('Container',(2.4,6.,2.6),(0,0,1.3))
        for x in (-1.21,1.21):
            for i in range(16):box('Rib',(.03,.055,2.5),(x,(i-7.5)*.36,1.3))
    elif kind in ('cardboard_box','crate','toolbox','storage_tub'):
        box('Body',(.6,.45,.4),(0,0,.2))
        if kind=='crate':
            for z in (.06,.34):
                for y in (-.23,.23):box('Slat',(.62,.035,.07),(0,y,z))
        elif kind=='toolbox':box('Lid',(.62,.47,.06),(0,0,.43));ring('Handle',(0,0,.52),.08,.015,axis='Y')
        elif kind=='cardboard_box':box('Tape',(.07,.455,.006),(0,0,.403),True)
    elif kind=='pallet':
        for i in range(6):box('Deck plank',(.13,1.2,.035),((i-2.5)*.17,0,.13))
        for y in (-.48,0,.48):box('Runner',(1.,.10,.10),(0,y,.065))
    elif kind in ('suitcase','backpack','handbag','briefcase','shopping_bag','duffel_bag'):
        w,d,h={'suitcase':(.45,.25,.65),'backpack':(.35,.20,.45),'handbag':(.35,.14,.25),'briefcase':(.44,.10,.30),'shopping_bag':(.30,.15,.40),'duffel_bag':(.6,.30,.28)}[kind]
        box('Bag',(w,d,h),(0,0,h/2));ring('Handle',(0,0,h+.04),min(.08,w/4),.012,axis='Y')
        if kind=='backpack':box('Pocket',(w*.7,.05,h*.4),(0,-d/2-.025,h*.35))
        if kind=='suitcase':wheels(w-.06,d-.04,.025)
    elif kind in ('oil_drum','gas_cylinder','jerry_can'):
        if kind=='jerry_can':box('Tank',(.35,.16,.45),(0,0,.225));ring('Grip',(0,0,.49),.065,.015,axis='Y')
        else:
            cyl('Tank',.28 if kind=='oil_drum' else .16,.85,(0,0,.425))
            if kind=='gas_cylinder':ellipsoid('Shoulder',(0,0,.85),(.32,.32,.18));cyl('Valve',.025,.12,(0,0,.97))
            else:
                for z in (.05,.78):ring('Hoop',(0,0,z),.28,.015)
    elif kind in ('palm_tree','deciduous_tree','hedge','cactus','flower','tree_stump','log','rock','boulder','flower_bed','pond','rock_wall'):
        if kind in ('rock','boulder'):ellipsoid('Rock',(0,0,.35),(1.2,.9,.7),subdivisions=1)
        elif kind in ('tree_stump','log'):cyl('Wood',.24,1.4 if kind=='log' else .45,(0,0,.24 if kind=='log' else .225),axis='Y' if kind=='log' else 'Z',vertices=10)
        elif kind=='cactus':
            cyl('Trunk',.13,1.4,(0,0,.7))
            for x,z in ((-.4,.8),(.35,1.0)):beam('Branch',(0,0,z),(x,0,z),.16);cyl('Tip',.08,.45,(x,0,z+.225))
        elif kind=='flower':
            beam('Stem',(0,0,0),(0,0,.55),.015)
            for i in range(6):a=i*math.tau/6;ellipsoid('Petal',(.08*math.cos(a),.08*math.sin(a),.55),(.12,.12,.04),1)
        elif kind=='hedge':box('Hedge',(2.,.65,1.1),(0,0,.55))
        elif kind=='flower_bed':
            box('Bed',(1.6,.8,.15),(0,0,.075))
            for x in (-.55,0,.55):ellipsoid('Plant',(x,0,.25),(.55,.65,.35),1)
        elif kind=='pond':ellipsoid('Water',(0,0,.04),(3.,2.,.08))
        elif kind=='rock_wall':
            for z in range(4):
                for x in range(5):ellipsoid('Stone',((x-2)*.4+(z%2)*.1,0,.2+z*.3),(.48,.38,.36),1)
        else:
            cyl('Trunk',.16,2.2,(0,0,1.1),vertices=8)
            if kind=='deciduous_tree':
                for x,y,z in ((0,0,2.4),(-.6,0,2.2),(.5,.2,2.4)):ellipsoid('Crown',(x,y,z),(1.5,1.35,1.3))
            else:
                for i in range(7):
                    a=i*math.tau/7;beam('Frond',(0,0,2.2),(1.2*math.cos(a),1.2*math.sin(a),1.8),.20,.025)
    elif kind=='fountain':cyl('Basin',.9,.16,(0,0,.08));cyl('Pedestal',.12,.9,(0,0,.6));cyl('Upper bowl',.48,.10,(0,0,1.1))
    elif kind in ('statue','mannequin','sculpture'):
        if kind=='sculpture':
            box('Plinth',(.6,.6,.2),(0,0,.1));ring('Abstract loop',(0,0,.9),.55,.13,axis='Y')
        else:
            box('Torso',(.4,.22,.5),(0,0,1.15));ellipsoid('Head',(0,0,1.62),(.24,.23,.30))
            for x in (-.13,.13):beam('Leg',(x,0,.9),(x,0,.05),.13)
            for x in (-.30,.30):beam('Arm',(x,0,1.35),(x,0,.85),.10)
            if kind=='statue':box('Plinth',(.65,.6,.15),(0,0,-.075))
    elif kind in ('picnic_table','park_bench'):
        if kind=='park_bench':table(1.7,.45,.48);box('Back',(1.7,.06,.45),(0,.2,.75))
        else:
            table(1.8,.75,.75)
            for x in (-.65,.65):box('Bench',(.32,1.8,.08),(x,0,.45))
    elif kind=='swing':
        for x in (-1.,1.):
            for y in (-.7,.7):beam('A-frame',(x,y,0),(x,0,2.2),.07)
        beam('Top bar',(-1,0,2.2),(1,0,2.2),.08);box('Seat',(.6,.25,.05),(0,0,.55))
        for x in (-.25,.25):beam('Chain',(x,0,.58),(x,0,2.2),.012)
    elif kind=='slide':
        ob=box('Slide',(.6,2.,.06),(0,0,.7));ob.rotation_euler.x=math.radians(35)
        for x in (-.28,.28):beam('Ladder rail',(x,.8,0),(x,.8,1.5),.04)
        for z in (.3,.6,.9,1.2):beam('Rung',(-.28,.8,z),(.28,.8,z),.03)
    elif kind=='seesaw':box('Fulcrum',(.3,.3,.5),(0,0,.25));ob=box('Board',(2.5,.25,.08),(0,0,.6));ob.rotation_euler.y=.15
    elif kind=='sandbox':
        box('Sand',(1.8,1.8,.08),(0,0,.04))
        for x in (-.95,.95):box('Rim',(.1,2.,.2),(x,0,.1))
        for y in (-.95,.95):box('Rim',(2.,.1,.2),(0,y,.1))
    elif kind=='tent':
        for x in (-.5,.5):ob=box('Canvas',(1.4,2.2,.035),(x,0,.55));ob.rotation_euler.y=math.radians(45 if x>0 else -45)
        box('Groundsheet',(2.,2.2,.02),(0,0,.01))
    elif kind in ('campfire','firewood'):
        for i in range(4):ob=cyl('Log',.10,1.,(0,0,.15+i*.07),axis='Y',vertices=8);ob.rotation_euler.z=i*.8
        if kind=='campfire':
            for x in (-.15,0,.15):cone('Flame',.15,0,.65,(x,0,.55))
    elif kind=='basketball_hoop':
        box('Base',(.8,.8,.1),(0,.3,.05));beam('Post',(0,.3,0),(0,.3,2.8),.08);box('Backboard',(1.0,.05,.65),(0,0,2.65));ring('Rim',(0,-.35,2.4),.22,.025)
    elif kind in ('soccer_goal','tennis_net'):
        h=1.0 if kind=='tennis_net' else 2.;w=3.
        for x in (-w/2,w/2):beam('Post',(x,0,0),(x,0,h),.05)
        beam('Top',(-w/2,0,h),(w/2,0,h),.05)
        if kind=='tennis_net':
            for i in range(14):beam('Net upright',(-w/2+i*w/13,0,.2),(-w/2+i*w/13,0,h),.008)
            for z in (.2,.4,.6,.8):beam('Net row',(-w/2,0,z),(w/2,0,z),.008)
        else:
            for x in (-w/2,w/2):beam('Back brace',(x,0,h),(x,.9,0),.035)
    elif kind=='pingpong_table':table(2.74,1.52,.76);box('Net',(2.74,.015,.15),(0,0,.87))
    elif kind in ('dumbbell','barbell'):
        length=1.8 if kind=='barbell' else .4;beam('Bar',(-length/2,0,.15),(length/2,0,.15),.03)
        for x in (-length*.38,length*.38):cyl('Weight',.18 if kind=='barbell' else .10,.08,(x,0,.15),axis='X',dark=True)
    elif kind=='weight_bench':table(1.1,.3,.45);box('Pad',(1.1,.32,.09),(0,0,.50))
    elif kind=='treadmill':
        box('Deck',(.75,1.65,.14),(0,0,.07));box('Belt',(.6,1.45,.012),(0,0,.146),True)
        for x in (-.33,.33):beam('Upright',(x,-.6,.14),(x,-.45,1.15),.045)
        box('Console',(.65,.20,.10),(0,-.45,1.15))
    elif kind=='exercise_bike':
        box('Foot',(.55,1.,.07),(0,0,.035));cyl('Flywheel',.25,.12,(0,-.2,.35),axis='X');beam('Seatpost',(0,.25,.1),(0,.25,.9),.05);box('Seat',(.25,.3,.06),(0,.25,.93));beam('Bars',(-.3,-.35,1.1),(.3,-.35,1.1),.035);beam('Stem',(0,-.3,.4),(0,-.35,1.1),.05)
    elif kind=='punching_bag':cyl('Bag',.22,1.0,(0,0,.65));beam('Strap',(0,0,1.15),(0,0,1.6),.03)
    elif kind in ('surfboard','snowboard','skateboard','skis'):
        length={'surfboard':1.9,'snowboard':1.5,'skateboard':.8,'skis':1.8}[kind]
        if kind=='skis':
            for x in (-.10,.10):box('Ski',(.10,length,.025),(x,0,.013));box('Binding',(.09,.3,.05),(x,0,.05))
        else:
            ellipsoid('Board',(0,0,.05),(.5 if kind=='surfboard' else .25,length,.07))
            if kind=='skateboard':wheels(.22,.5,.035)
    elif kind=='tennis_racket':ring('Head',(0,0,.4),.15,.012,axis='Y');beam('Handle',(0,0,0),(0,0,.28),.025)
    elif kind=='baseball_bat':cone('Bat',.018,.04,.85,(0,0,.425))
    elif kind in ('football','basketball','tennis_ball','ball','egg','apple','orange'):
        size={'football':(.22,.34,.22),'basketball':(.24,.24,.24),'tennis_ball':(.067,.067,.067),'ball':(.2,.2,.2),'egg':(.045,.045,.065),'apple':(.08,.08,.075),'orange':(.08,.08,.08)}[kind]
        ellipsoid(kind,(0,0,size[2]/2),size)
        if kind=='apple':beam('Stem',(0,0,size[2]),(.005,0,size[2]+.02),.006)
    elif kind=='helmet':ellipsoid('Shell',(0,0,.12),(.26,.30,.24));box('Visor',(.24,.035,.08),(0,-.15,.13),True)
    elif kind=='guitar':ellipsoid('Body',(0,0,.3),(.38,.12,.48));beam('Neck',(0,0,.5),(0,0,1.0),.065,.035);box('Headstock',(.08,.04,.14),(0,0,1.05))
    elif kind=='piano':
        cabinet(1.4,1.2,.55);box('Keyboard ledge',(1.4,.3,.10),(0,-.38,.75))
        for i in range(24):box('Key',(.045,.20,.02),((i-11.5)*.05,-.42,.815),i%3==0)
    elif kind=='drum':cyl('Drum',.3,.35,(0,0,.30));
    elif kind in ('music_stand','microphone_stand','iv_stand'):
        cyl('Pole',.018,1.5,(0,0,.75))
        for i in range(3):a=i*math.tau/3;beam('Foot',(0,0,.1),(.3*math.cos(a),.3*math.sin(a),.02),.025)
        if kind=='music_stand':ob=box('Score rest',(.5,.035,.35),(0,0,1.5));ob.rotation_euler.x=-.3
        elif kind=='microphone_stand':beam('Boom',(0,0,1.5),(0,-.6,1.7),.02)
        else:beam('Hook bar',(-.2,0,1.6),(.2,0,1.6),.015)
    elif kind=='microphone':cyl('Grip',.025,.18,(0,0,.09));ellipsoid('Head',(0,0,.21),(.075,.075,.08))
    elif kind in ('toy_block','chessboard'):box('Block' if kind=='toy_block' else 'Board',(.12,.12,.12) if kind=='toy_block' else (.4,.4,.02),(0,0,.06 if kind=='toy_block' else .01))
    elif kind=='teddy_bear':
        ellipsoid('Body',(0,0,.22),(.25,.20,.3));ellipsoid('Head',(0,-.015,.42),(.23,.21,.22))
        for x in (-.10,.10):ellipsoid('Ear',(x,0,.51),(.08,.06,.09));ellipsoid('Foot',(x,-.03,.07),(.10,.14,.12));ellipsoid('Arm',(x,0,.25),(.10,.09,.2))
    elif kind in ('hammer','handsaw','wrench','paint_brush'):
        beam('Handle',(0,0,0),(0,0,.28),.025)
        if kind=='hammer':box('Head',(.18,.045,.045),(0,0,.28))
        elif kind=='handsaw':box('Blade',(.25,.012,.12),(.10,0,.22))
        elif kind=='wrench':ring('Jaw',(0,0,.3),.045,.012,axis='Y')
        else:box('Brush',(.07,.025,.08),(0,0,.30))
    elif kind in ('generator','air_compressor'):
        box('Base',(.7,.45,.08),(0,0,.04));wheels(.6,.32,.08)
        if kind=='generator':box('Engine',(.45,.34,.35),(0,0,.27));box('Tank',(.55,.38,.12),(0,0,.50))
        else:cyl('Tank',.20,.65,(0,0,.30),axis='X');box('Motor',(.3,.2,.2),(0,0,.6))
    elif kind=='work_light':cyl('Stand',.025,1.3,(0,0,.65));box('Lamp',(.35,.1,.25),(0,0,1.35));box('Base',(.5,.5,.04),(0,0,.02))
    elif kind=='shopping_display':shelf(1.,1.6,.45,4)
    elif kind=='microscope':box('Base',(.25,.3,.03),(0,0,.015));beam('Arm',(0,.08,.03),(0,.08,.35),.05);box('Stage',(.18,.18,.025),(0,0,.18));beam('Optics',(0,0,.27),(0,.05,.43),.045)
    elif kind in ('dog','cat','horse','cow','sheep'):
        animal({'dog':(.28,.65,.55),'cat':(.20,.42,.32),'horse':(.55,1.5,1.55),'cow':(.7,1.6,1.35),'sheep':(.5,.85,.75)}[kind])
    elif kind in ('bird','duck','fish'):
        ellipsoid('Body',(0,0,.13),(.18,.35,.22));ellipsoid('Head',(0,-.16,.24),(.13,.13,.13))
        ob=cone('Beak',.035,0,.09,(0,-.23,.24));ob.rotation_euler.x=math.pi/2
        if kind=='fish':box('Tail',(.18,.018,.16),(0,.22,.13))
        else:
            for x in (-.04,.04):beam('Leg',(x,0,.1),(x,0,0),.012)
    elif kind=='banana':
        # One closed curved mesh rather than a pile of disconnected cylinders.
        vertices=[];faces=[];sections=9;sides=8
        for i in range(sections):
            t=i/(sections-1);x=(t-.5)*.22;z=.035+.12*(t-.5)**2
            radius=.008+.018*math.sin(math.pi*t)
            for j in range(sides):
                angle=j*math.tau/sides;vertices.append((x,radius*math.cos(angle),z+radius*math.sin(angle)))
        for i in range(sections-1):
            for j in range(sides):
                a=i*sides+j;b=i*sides+(j+1)%sides;faces.append((a,b,b+sides,a+sides))
        faces.extend([tuple(reversed(range(sides))),tuple((sections-1)*sides+j for j in range(sides))])
        mesh=bpy.data.meshes.new('Banana');mesh.from_pydata(vertices,[],faces);mesh.update()
        ob=bpy.data.objects.new('Fruit',mesh);bpy.context.collection.objects.link(ob);ob.data.materials.append(material())
    elif kind=='bread':ellipsoid('Loaf',(0,0,.1),(.35,.18,.20))
    elif kind=='cake':cyl('Cake',.16,.12,(0,0,.06));cyl('Icing',.17,.015,(0,0,.1275))
    elif kind=='pizza':cyl('Pizza',.18,.018,(0,0,.009))
    else:raise ValueError('No placeholder geometry for '+kind)
