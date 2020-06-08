import pandas as pd
import SNN

def index_2d (list_2d, element):
    for row, row_list in enumerate(list_2d):
        if element in row_list:
            return (row, row_list.index(element))

def index_2d_multirows (list_2d, element):
    row_indices = []
    indices_in_row_list = []
    for row, row_list in enumerate(list_2d):
        if element in row_list:
            row_indices.append(row)
            indices_in_row_list.append(row_list.index(element))
    if len(indices_in_row_list) != len(row_indices):
        print("Warning when calling index_2d_multirows(): multiple occurances in row index {} of element {}"
            .format(row, element))
    return (row_indices, indices_in_row_list)

def initializeNetWorkConnectivity(num_categories, num_edge_maps, W_input, F_hidden, S_hidden,
                                  depth_hidden_per_sublocation, weight_vector, sheet_dir):
    num_input_neurons = W_input**2 * num_edge_maps 

    W_hidden = \
        int((W_input-F_hidden) / S_hidden) + 1

    # num_hidden_neurons_per_slice is the number of sublocations in the input spatial grid
    num_hidden_neurons_per_slice = W_hidden**2
    num_hidden_neurons = num_hidden_neurons_per_slice * depth_hidden_per_sublocation
    num_output_neurons = num_categories
    num_neurons = num_input_neurons + num_hidden_neurons + num_output_neurons
#%% Establish Connectivity Information
#############################################################################################
    # a list of indicies for output layer neurons
    output_neuron_list = [idx for idx in range(num_input_neurons+num_hidden_neurons, num_input_neurons+num_hidden_neurons+num_categories)]

    # a list of indices for hidden layer neurons
    hidden_neuron_list_flattened = [idx for idx in range(num_input_neurons, num_input_neurons+num_hidden_neurons)]

    # a list of indices for input layer neurons
    input_neuron_list = [idx for idx in range(0,num_input_neurons,1)]

###################### Establish Hidden Layer Connectivity ##################
    ## a list of indicies for hidden layer neurons
    ## hidden_neuron_list[sublocation_idx][depth_idx]
    hidden_neuron_list = [
                            [None for depth_idx in range(depth_hidden_per_sublocation)] 
                            for sublocation_idx in range(num_hidden_neurons_per_slice)
                        ]

    for sublocation_idx in range(num_hidden_neurons_per_slice):
        for depth_idx in range(depth_hidden_per_sublocation):
            hidden_neuron_list[sublocation_idx][depth_idx] = \
                num_input_neurons + sublocation_idx * depth_hidden_per_sublocation + depth_idx
    
    ## specify pixel indices in the receptive field of each hidden layer neuron within one depth slice
    ## based on the row-major movement of convolutional kernel
    receptive_pixel_indices = \
        [  
            [None for elements in range(F_hidden**2)]
            for sublocation_idx in range(num_hidden_neurons_per_slice)
        ]
    for i in range(num_hidden_neurons_per_slice):
        starting_pixel_idx = (i // W_hidden) * W_input * S_hidden + (i % W_hidden) * S_hidden
        for element_idx in range(F_hidden**2):
            receptive_pixel_indices[i][element_idx] = \
                (element_idx // F_hidden)*W_input + (element_idx % F_hidden) + starting_pixel_idx

    ## from the pixel indices in the receptive field of each hidden layer neuron, specify the 
    ## presynaptic input neuron index for each hidden layer neuron
    ## and fan-in synapse addresses for each hidden layer neuron
    presynaptic_input_neuron_indices = \
        [
            [
                [None for input_neuron_idx in range((F_hidden**2) * num_edge_maps)]
                for depth_idx in range(depth_hidden_per_sublocation)
            ] for sublocation_idx in range(num_hidden_neurons_per_slice)
        ]

    fan_in_synapse_addrs = \
        [
            [
                [None for synapse_idx in range((F_hidden**2)*num_edge_maps)]
                for depth_idx in range(depth_hidden_per_sublocation)
            ] for sublocation_idx in range(num_hidden_neurons_per_slice)
        ]     

    for sublocation_idx in range(num_hidden_neurons_per_slice):
        for depth_idx in range(depth_hidden_per_sublocation):
            for edge_map_idx in range(num_edge_maps):
                presynaptic_input_neuron_indices[sublocation_idx][depth_idx][edge_map_idx*F_hidden**2 : (edge_map_idx+1)*F_hidden**2] = \
                    list(map(lambda x: edge_map_idx*W_input**2 + x, receptive_pixel_indices[sublocation_idx]))
                fan_in_synapse_addrs[sublocation_idx][depth_idx][edge_map_idx*F_hidden**2 : (edge_map_idx+1)*F_hidden**2] = \
                    list(map(lambda x: ((sublocation_idx*depth_hidden_per_sublocation + depth_idx) * num_edge_maps + edge_map_idx) * (F_hidden**2) + num_input_neurons + x, 
                            [element for element in range(F_hidden**2)]))

    ## flatten presynaptic_input_neuron_indices and fan_in_synapse_addrs
    ## indexing them using hidden neuron indices and record the connectivity
    ## in a list of dictionaries
    hidden_connectivity = \
        [
            {
                "neuron_idx"            :   None,
                "depth_idx"             :   None, 
                "sublocation_idx"       :   None,
                "fan_in_synapse_addrs"  :   None,
                "fan_in_neuron_indices" :   None,
                "fan_out_synapse_indices":   None,
                "fan_out_neuron_indices":   None
            } for hidden_neuron in range(num_hidden_neurons) 
        ] 
    first_fan_in_synapse_output = num_input_neurons + num_hidden_neurons * F_hidden**2 * num_edge_maps
    for idx in range(num_hidden_neurons):
        neuron_idx = num_input_neurons + idx
        hidden_connectivity[idx]["neuron_idx"] = neuron_idx
        
        sublocation_idx, depth_idx = divmod(idx, depth_hidden_per_sublocation)
        hidden_connectivity[idx]["depth_idx"] = depth_idx
        hidden_connectivity[idx]["sublocation_idx"] = sublocation_idx

        fan_in_synapses = fan_in_synapse_addrs[sublocation_idx][depth_idx]
        hidden_connectivity[idx]["fan_in_synapse_addrs"] = fan_in_synapses
        
        fan_in_neurons = presynaptic_input_neuron_indices[sublocation_idx][depth_idx]
        hidden_connectivity[idx]["fan_in_neuron_indices"] = fan_in_neurons

        hidden_connectivity[idx]["fan_out_neuron_indices"] = output_neuron_list

        hidden_connectivity[idx]["fan_out_synapse_indices"] = \
            list(map(lambda x: first_fan_in_synapse_output + idx + x, [group_idx for group_idx in range(0, num_output_neurons*num_hidden_neurons, num_hidden_neurons)]))
##########################################################################
                    
###################### Establish Input Layer Connectivity ##################
    input_connectivity = \
        [
            {
                "neuron_idx"            :   None,
                "edge_map_idx"          :   None,
                "pixel_idx"             :   None,
                "receptive_field_info"  :   {"sublocation_idx" : [], "kernel_element_idx" : []},
                "fan_in_synapse_addrs"  :   [],
                "fan_in_neuron_indices" :   [],
                "fan_out_synapse_indices":  [],
                "fan_out_neuron_indices":   []
            } for input_neuron in range(num_input_neurons) 
        ] 
    for idx in range(num_input_neurons):
        edge_map_idx, pixel_idx = divmod(idx, W_input**2)
        input_connectivity[idx]["neuron_idx"] = [idx]
        input_connectivity[idx]["edge_map_idx"] = edge_map_idx
        input_connectivity[idx]["pixel_idx"] = pixel_idx
        input_connectivity[idx]["fan_in_synapse_addrs"] = [idx]
        input_connectivity[idx]["fan_in_neuron_addrs"] = []
        
        # find out which sublocation's receptive field this pixel belongs to
        sublocation_idx_list, kernel_element_idx_list = \
            index_2d_multirows(receptive_pixel_indices, pixel_idx)
        # for each sublocation_idx, traverse all the slices
        for i in range(len(sublocation_idx_list)):
            sublocation_idx = sublocation_idx_list[i]
            kernel_element_idx = kernel_element_idx_list[i]  
            input_connectivity[idx]["receptive_field_info"]["sublocation_idx"].append(sublocation_idx)
            input_connectivity[idx]["receptive_field_info"]["kernel_element_idx"].append(kernel_element_idx)


            fan_out_neuron_list = hidden_neuron_list[sublocation_idx]
            input_connectivity[idx]["fan_out_neuron_indices"].extend(fan_out_neuron_list)

            # with each fan-out neuron, there is a fan-out synapse to be connected
            for fan_out_neuron_idx in fan_out_neuron_list:
                j = fan_out_neuron_idx - num_input_neurons
                depth_idx = hidden_connectivity[j]["depth_idx"]
                fan_out_synapse_idx = \
                    num_input_neurons + ((((sublocation_idx * depth_hidden_per_sublocation) + depth_idx) * num_edge_maps) + edge_map_idx) * F_hidden**2 + kernel_element_idx
                input_connectivity[idx]["fan_out_synapse_indices"].append(fan_out_synapse_idx)

    # # check if fan_out_neuron_indices and fan_out_synapse_indices are recorded by 
    # # intended hidden-layer neuron
    # for input_neuron_idx in input_neuron_list:
    #     for fan_out_neuron_idx in input_connectivity[input_neuron_idx]["fan_out_neuron_indices"]:
    #         i = input_connectivity[input_neuron_idx]["fan_out_neuron_indices"].index(fan_out_neuron_idx)
    #         j = fan_out_neuron_idx - num_input_neurons
    #         if not input_neuron_idx in hidden_connectivity[j]["fan_in_neuron_indices"]:
    #             print("input neuron {} is not contained in the fan-in neuron of hidden neuron {}".format(input_neuron_idx, fan_out_neuron_idx)) 
    #         synapse_idx =  input_connectivity[input_neuron_idx]["fan_out_synapse_indices"][i]
    #         if not synapse_idx in hidden_connectivity[j]["fan_in_synapse_addrs"]:
    #             print("fan-out synapse {} of input neuron {} is not recorded in the fan-in synapse of hidden neuron {}".format(synapse_idx, input_neuron_idx, fan_out_neuron_idx)) 
#############################################################################

###################### Establish Output Layer Connectivity ##################
    ## Output neurons are fully-connected to the hidden layer
    output_connectivity = \
        [
            {
                "neuron_idx"            :   None,
                "category_idx"          :   None,
                "fan_in_synapse_addrs"  :   [],
                "fan_in_neuron_indices" :   [],
                "fan_out_synapse_indices":  [],
                "fan_out_neuron_indices":   []
            } for input_neuron in range(num_output_neurons) 
        ] 
    first_neuron_idx_output = num_input_neurons + num_output_neurons
    for idx in range(num_output_neurons):
        output_connectivity[idx]["neuron_idx"] = idx + first_neuron_idx_output
        output_connectivity[idx]["category_idx"] = idx
        output_connectivity[idx]["fan_in_neuron_indices"] = hidden_neuron_list_flattened
        output_connectivity[idx]["fan_in_synapse_addrs"] = \
            [
                synapse_idx for synapse_idx in range(first_fan_in_synapse_output + idx * num_hidden_neurons, 
                    first_fan_in_synapse_output + (idx + 1) * num_hidden_neurons)
            ] 

###################### Initialize Connectivity Table ##################
    ConnectivityTable = SNN.ConnectivityInfo(num_neurons=num_neurons)
    for i in range(num_input_neurons):
        neuron_idx = i
        ConnectivityTable.layer_num[neuron_idx] = 0
        ConnectivityTable.fan_in_neuron_idx[neuron_idx] = input_connectivity[i]["fan_in_neuron_indices"]
        ConnectivityTable.fan_in_synapse_addr[neuron_idx] = input_connectivity[i]["fan_in_synapse_addrs"]
        ConnectivityTable.fan_out_neuron_idx[neuron_idx] = input_connectivity[i]["fan_out_neuron_indices"]
        ConnectivityTable.fan_out_synapse_addr[neuron_idx] = input_connectivity[i]["fan_out_synapse_indices"]

    for i in range(num_hidden_neurons):
        neuron_idx = num_input_neurons + i
        ConnectivityTable.layer_num[neuron_idx] = 1
        ConnectivityTable.fan_in_neuron_idx[neuron_idx] = hidden_connectivity[i]["fan_in_neuron_indices"]
        ConnectivityTable.fan_in_synapse_addr[neuron_idx] = hidden_connectivity[i]["fan_in_synapse_addrs"]
        ConnectivityTable.fan_out_neuron_idx[neuron_idx] = hidden_connectivity[i]["fan_out_neuron_indices"]
        ConnectivityTable.fan_out_synapse_addr[neuron_idx] = hidden_connectivity[i]["fan_out_synapse_indices"]

    for i in range(num_output_neurons):
        neuron_idx = num_input_neurons + num_hidden_neurons + i
        ConnectivityTable.layer_num[neuron_idx] = 2
        ConnectivityTable.fan_in_neuron_idx[neuron_idx] = output_connectivity[i]["fan_in_neuron_indices"]
        ConnectivityTable.fan_in_synapse_addr[neuron_idx] = output_connectivity[i]["fan_in_synapse_addrs"]
        ConnectivityTable.fan_out_neuron_idx[neuron_idx] = output_connectivity[i]["fan_out_neuron_indices"]
        ConnectivityTable.fan_out_synapse_addr[neuron_idx] = output_connectivity[i]["fan_out_synapse_indices"]
    
#############################################################################################

###################### Initialize WeightRAM #########################
    num_synapses = output_connectivity[-1]["fan_in_synapse_addrs"][-1] + 1 
    WeightRAM = SNN.WeightRAM(num_synapses=num_synapses)
    for synapse_addr in range(num_synapses):
        post_neuron_idx_WRAM, _ = index_2d(ConnectivityTable.fan_in_synapse_addr, synapse_addr)
        WeightRAM.post_neuron_idx[synapse_addr] = post_neuron_idx_WRAM
        WeightRAM.weight[synapse_addr] = weight_vector[synapse_addr]
        if synapse_addr >= num_input_neurons:
            pre_neuron_idx_WRAM, _ = index_2d(ConnectivityTable.fan_out_synapse_addr, synapse_addr)
            WeightRAM.pre_neuron_idx[synapse_addr] = pre_neuron_idx_WRAM
	
        # fields for training statistics
        WeightRAM.post_neuron_layer[synapse_addr] = ConnectivityTable.layer_num[post_neuron_idx_WRAM]

        if ConnectivityTable.layer_num[post_neuron_idx_WRAM] == 0:
            WeightRAM.post_neuron_location[synapse_addr] = \
                    input_connectivity[post_neuron_idx_WRAM]["pixel_idx"]
        elif ConnectivityTable.layer_num[post_neuron_idx_WRAM] == 1:
            WeightRAM.post_neuron_location[synapse_addr] = \
                    hidden_connectivity[post_neuron_idx_WRAM-num_input_neurons]["sublocation_idx"]
        elif ConnectivityTable.layer_num[post_neuron_idx_WRAM] == 2:
            WeightRAM.post_neuron_location[synapse_addr] = \
                    output_connectivity[post_neuron_idx_WRAM-num_input_neurons-num_hidden_neurons]["category_idx"]

#############################################################################################


###################### Initialize PotentialRAM #########################
    PotentialRAM = SNN.PotentialRAM(num_neurons=num_neurons)
    PotentialRAM.fan_out_synapse_addr = ConnectivityTable.fan_out_synapse_addr
#############################################################################################

#%% write connectivity info to spreadsheet
    input_layer_df = \
        pd.DataFrame(
            {
                'Layer Index'       :   [0] * num_input_neurons,  
                'Neuron Index'      :   input_neuron_list,
                'Edge Map Index'    :   [input_connectivity[i]["edge_map_idx"] for i in range(num_input_neurons)],
                'Sublocation Index' :   [input_connectivity[i]["receptive_field_info"]["sublocation_idx"] for i in range(num_input_neurons)],
                'Kernel Element Index' : [input_connectivity[i]["receptive_field_info"]["kernel_element_idx"] for i in range(num_input_neurons)],
                'Fan-in Neuron Indices':  [None for i in range(num_input_neurons)],
                'Fan-in Synapse Indices': [input_connectivity[i]["fan_in_synapse_addrs"] for i in range(num_input_neurons)],
                'Fan-out Neuron Indices': [input_connectivity[i]["fan_out_neuron_indices"] for i in range(num_input_neurons)],
                'Fan-out Synapse Indices': [input_connectivity[i]["fan_out_synapse_indices"] for i in range(num_input_neurons)]
            
            }
        )
    input_layer_df.name = 'Input Layer'
    
    hidden_layer_df = \
        pd.DataFrame(
            {
                'Layer Index'       :   [1] * num_hidden_neurons,
                'Neuron Index'      :   hidden_neuron_list_flattened,
                'Depth Index'       :   [hidden_connectivity[i]["depth_idx"] for i in range(num_hidden_neurons)],
                'Sublocation Index' :   [hidden_connectivity[i]["sublocation_idx"] for i in range(num_hidden_neurons)],
                'Null Field0'       :   [None] * num_hidden_neurons,
                'Fan-in Neuron Indices':  [hidden_connectivity[i]["fan_in_neuron_indices"] for i in range(num_hidden_neurons)],
                'Fan-in Synapse Indices': [hidden_connectivity[i]["fan_in_synapse_addrs"] for i in range(num_hidden_neurons)],
                'Fan-out Neuron Indices': [hidden_connectivity[i]["fan_out_neuron_indices"] for i in range(num_hidden_neurons)],
                'Fan-out Synapse Indices': [hidden_connectivity[i]["fan_out_synapse_indices"] for i in range(num_hidden_neurons)]
            }   
        )
    hidden_layer_df.name = "Hidden Layer"

    output_layer_df = \
        pd.DataFrame(
            {
                'Layer Index'       :   [2] * num_output_neurons,
                'Neuron Index'      :   output_neuron_list,
                'Category Index'    :   [output_connectivity[i]["category_idx"] for i in range(num_output_neurons)],
                'Null Field1'       :   [None] * num_output_neurons,
                'Null Field2'       :   [None] * num_output_neurons,
                'Fan-in Neuron Indices':  [output_connectivity[i]["fan_in_neuron_indices"] for i in range(num_output_neurons)],
                'Fan-in Synapse Indices': [output_connectivity[i]["fan_in_synapse_addrs"] for i in range(num_output_neurons)],
                'Fan-out Neuron Indices': [None] * num_output_neurons,
                'Fan-out Synapse Indices': [None] * num_output_neurons
            }
        )
    output_layer_df.name="Output Layer"

    synapse_connectivity_df = \
        pd.DataFrame(
            {
                'Synapse Index'         :   [i for i in range(num_synapses)],
                'Presynpatic Neuron'    :   WeightRAM.pre_neuron_idx,
                'Postsynaptic Neuron'   :   WeightRAM.post_neuron_idx
            }
        )



    writer = pd.ExcelWriter(sheet_dir, engine='xlsxwriter')
    workbook = writer.book
    merge_format = workbook.add_format({
        'bold'  : True,
        'align' : 'center'
                                        })
    cell_format = workbook.add_format({'align':'center'})

    input_layer_df.to_excel(writer, sheet_name='Sheet1', index=False, startrow=0, startcol=0)
    hidden_layer_df.to_excel(writer, sheet_name='Sheet1', index=False, startrow=input_layer_df.shape[0]+5, startcol=0)
    output_layer_df.to_excel(writer, sheet_name='Sheet1', index=False, startrow=input_layer_df.shape[0]+5+hidden_layer_df.shape[0]+4, startcol=0)
    worksheet1 = writer.sheets['Sheet1']
    # worksheet.merge_range('A1:I1', input_layer_df.name, merge_format)
    # worksheet.merge_range('A69:I69', hidden_layer_df.name, merge_format)
    # worksheet.merge_range('A119:I119', hidden_layer_df.name, merge_format)
    worksheet1.set_column('A:K', 20, cell_format)
    

    synapse_connectivity_df.to_excel(writer, sheet_name='Sheet2', index=False, startrow=0, startcol=0)
    worksheet2 = writer.sheets['Sheet2']
    worksheet2.set_column('A:C', 20, cell_format)
  
  
    writer.save()



    return(input_connectivity, hidden_connectivity, output_connectivity, ConnectivityTable, WeightRAM, PotentialRAM, writer)                        


