#Optimisations : 

#1.  Sigmoid and Tanh activations threshold implementation.

#Shashwat torch imports
import torc
from torch import nn

class QuantReluHandler(QuantActBaseHandler):
    """Class for converting a quantized relu operation expressed in the QONNX
    dialect to the FINN ONNX dialect."""

    @classmethod
    def valid_predecessor_op_types(self):
        return [
            "Relu",
            "Selu",
            "Sigmoid", #Shashwat
            "Tanh", #Shashwat
        ]

class QuantReluHandler(QuantActBaseHandler):
    """Class for converting a quantized relu operation expressed in the QONNX
    dialect to the FINN ONNX dialect."""

    @classmethod
    def valid_predecessor_op_types(self):
        return [
            "Relu",
            "Selu",
            "Sigmoid", #Shashwat
            "Tanh", #Shashwat
        ]

    def _check_compatibility(self):
        if self._q_node.op_type == "Quant":
            q_inst = getCustomOp(self._q_node)
            narrow = q_inst.get_nodeattr("narrow")
            signed = q_inst.get_nodeattr("signed")
            if not self._model.get_initializer(self._q_node.input[2]) == 0:
                raise ValueError(
                    "Only Quant nodes with zero-point == 0 "
                    "are currently supported for ReLu activations."
                )
            act_node = self._model.find_direct_predecessors(self._q_node)
            act_node = act_node[0]
            if act_node.op_type == "Relu":
                if signed or narrow:
                    raise ValueError(
                        "FINN only supports unsigned and non-narrow Quant nodes "
                        "for Relu activations."
                    )
        elif self._q_node.op_type == "BipolarQuant":
            return
        else:
            raise RuntimeError("Got an unexpected quantizer node type")

    def _calculate_act_bias(self):
        # No bias allowed for Relu activations, see: https://github.com/Xilinx/
        # brevitas/blob/a5bfd6dc5e030f0047ac1ee47932b60e8e873e17/src/brevitas/
        # export/onnx/finn/handler/act.py#L48
        act_node = self._model.find_direct_predecessors(self._q_node)
        act_node = act_node[0]
        if act_node.op_type == "Relu":
            bias = np.array([0.0], dtype=np_default_dtype)
        elif act_node.op_type == "Selu":
            # Gather parameters
            q_inst = getCustomOp(self._q_node)
            if self._q_node.op_type == "Quant":
                bit_width = self._model.get_initializer(self._q_node.input[3])
                narrow = q_inst.get_nodeattr("narrow")
            elif self._q_node.op_type == "BipolarQuant":
                bit_width = 1.0
            else:
                raise RuntimeError("Got an unexpected quantizer node type")
            # Calculate bias, see: https://github.com/Xilinx/brevitas/blob/
            # a5bfd6dc5e030f0047ac1ee47932b60e8e873e17/src/brevitas/export/
            # onnx/finn/handler/act.py#L64
            if bit_width == 1.0:
                bias = np.array([-0.5], dtype=np_default_dtype)
            else:
                if narrow:
                    min_non_scaled_val = -(2 ** (bit_width - 1) - 1)
                else:
                    min_non_scaled_val = -(2 ** (bit_width - 1))
                bias = np.array([min_non_scaled_val], dtype=np_default_dtype)
        #This elif from Shashwat : Should be the same for both Sigmoid and Tanh according to my implementation 
        elif act_node.op_type == "Sigmoid":
            q_inst = getCustomOp(self._q_node)
            if self._q_node.op_type == "Quant":
                bit_width = self._model.get_initializer(self._q_node.input[3])
                narrow = q_inst.get_nodeattr("narrow")
            elif self._q_node.op_type == "BipolarQuant":
                bit_width = 1.0
            else:
                raise RuntimeError("Got an unexpected quantizer node type")
            
            if bit_width == 1.0:
                bias = np.array([0], dtype=np_default_dtype)   #For now set to '0' have to figure out what the actual value is.
            else:
                if narrow:
                    min_non_scaled_val = 0
                else:
                    min_non_scaled_val = 0
                bias = np.array([min_non_scaled_val], dtype=np_default_dtype)
                #bias value needs to be -128 or -127 for 8 bit activation to bring the output in the INT8 range. Tried 0 and that did not work for me
        
        elif act_node.op_type == "Tanh":
            q_inst = getCustomOp(self._q_node)
            if self._q_node.op_type == "Quant":
                bit_width = self._model.get_initializer(self._q_node.input[3])
                narrow = q_inst.get_nodeattr("narrow")
            elif self._q_node.op_type == "BipolarQuant":
                bit_width = 1.0
            else:
                raise RuntimeError("Got an unexpected quantizer node type")
            
            if bit_width == 1.0:
                bias = np.array([0], dtype=np_default_dtype)   #For now set to '0' have to figure out what the actual value is.
            else:
                if narrow:
                    min_non_scaled_val = -(2 ** (bit_width - 1) - 1)
                else:
                    min_non_scaled_val = -(2 ** (bit_width - 1)) 
                bias = np.array([min_non_scaled_val], dtype=np_default_dtype)
                #bias value needs to be -128 or -127 for 8 bit activation to bring the output in the INT8 range. Tried 0 and that did not work for me


        return bias

elif act_node.op_type == "Sigmoid" :
            q_inst = getCustomOp(self._q_node)
            flag = 0
            # max_range = 0
            # min_range = 0
            narrow = q_inst.get_nodeattr("narrow")
            if narrow:
                num_distinct_values = 2 ** bit_width - 1
                start_val = -(2 ** (bit_width - 1) - 1)
            else:
                num_distinct_values = 2 ** bit_width
                start_val = -(2 ** (bit_width - 1))
            
            if(bit_width == 8):
                max_range = (1/quant_scale) #(This I used for INT8 quantization with precomputed scales)
            if(bit_width == 6 and quant_scale == 0.015187746845185757 ):#Post-activation quantizer scale
                max_range = 31*0.10569057613611221             #Pre-activation quantizer scale
                min_range = -31*0.10569057613611221              #Pre-activation quantizer scale
                # max_range = round(max_range)
                # min_range = round(min_range)
                print("Sigmoid1",quant_scale)
            if(bit_width == 6 and quant_scale == 0.015358800999820232): #Post-activation quantizer scale
                max_range = 31*0.11701396107673645               #Pre-activation quantizer scale
                min_range = -31*0.11701396107673645             #Pre-activation quantizer scale
                # max_range = round(max_range)
                # min_range = round(min_range)
                print("Sigmoid2",quant_scale)
            if(bit_width == 6 and quant_scale == 0.014742395840585232): #Post-activation quantizer scale
                max_range = 31*0.09088636934757233            #Pre-activation quantizer scale
                min_range = -31*0.09088636934757233             #Pre-activation quantizer scale
                # max_range = round(max_range)
                # min_range = round(min_range)
                print("Sigmoid3",quant_scale)   

            # max_range = round(max_range)
            #Defining inputs that the sigmoid activation will see
            if(max_range == 128 and bit_width == 8):
                int_input_range = np.linspace(-1.002, 0.999, num=255)#Had to set this to -0.502 because the first input that the MT node got was -0.5019 which was less than -0.501 that I had set earlier.
            if(max_range == 255 and bit_width == 8):
                int_input_range = np.linspace(-0.502, 0.499, num=255)
            if(bit_width == 6):
                int_input_range = np.linspace(min_range,max_range,num=63) #63 thresholds for 6-bit quantization

            int_input_range = torch.from_numpy(int_input_range) #conversion to torch tensor
            sigmoid_out = nn.Sigmoid()              #defining sigmoid activation
            output = sigmoid_out(int_input_range)   #output of the activation function)

            #QuantizeLinear operation
            zero_point = 0 #For 8 bit activation Since the output is unsigned INT
            output = torch.round(output/quant_scale)
            #Factoring in the zero point
            output = output + zero_point 

            #Need to clip the outputs here to repliacte the fucnctioning of the Quant Node.
            #Clamping operation
            min_int_val = 0
            max_int_val = 63
            output = np.where(output > max_int_val,max_int_val, output)
            output = np.where(output < min_int_val,min_int_val, output)

            unique_input_indices = np.unique(output, return_index=True)[1] #[1:] #These are inputs which cause a change in the levels of the integer outputs from the quantization function
            unique_inputs = np.zeros(len(unique_input_indices))
            unique_inputs = int_input_range[unique_input_indices]#[1:] #Did not need to use [1:] as I am computing thresholds for 255 inputs only #identifying these inputs from the input_range array and ignoring the first threshold
            acivation_bit_width = bit_width
            num_thresholds = int(2 ** acivation_bit_width - 1)
            thresholds = np.zeros(num_thresholds)

            threshold_index = 0
            output_index = 0
            index_pos = 0
            while threshold_index < 63 and output_index < 62 and index_pos < len(unique_input_indices):
                if output[output_index] == output[output_index+1]: 
                    output_index += 1
                elif output[output_index] != output[output_index+1]:
                    if(index_pos == 0):
                        diff = output[0]
                        # index_pos = index_pos+1
                        # continue
                    elif(index_pos != 0):
                        diff = output[output_index+1] - output[output_index] # Calculating number of required repeats
                    while diff > 0:  #Copying repeats into the threshold thershold matrix       
                        thresholds[threshold_index] = unique_inputs[index_pos] 
                        threshold_index += 1
                        diff -= 1
                    output_index += 1
                    index_pos += 1
            
            for i in range(threshold_index, 63):
                if(index_pos != len(unique_input_indices)):
                    thresholds[i] = (int_input_range[unique_input_indices[index_pos]])
                else:
                    thresholds[i] = max_range+1 

        elif act_node.op_type == "Tanh" :
            q_inst = getCustomOp(self._q_node)
            narrow = q_inst.get_nodeattr("narrow")
            if narrow:
                num_distinct_values = 2 ** bit_width - 1
                start_val = -(2 ** (bit_width - 1) - 1)
            else:
                num_distinct_values = 2 ** bit_width
                start_val = -(2 ** (bit_width - 1))
            
            if(bit_width == 8):
                max_range = (1/quant_scale)
                max_range = round(max_range)
            if(bit_width==6 and quant_scale == 0.03136685863137245):#Post-activation quantizer scale
                max_range = 31*0.0793924629688263   #Pre-activation quantizer scale
                min_range = -31*0.0793924629688263  #Pre-activation quantizer scale
                # max_range = round(max_range)
                # min_range = round(min_range)
                print("Tanh1",quant_scale)
            if(bit_width==6 and quant_scale == 0.02342350408434868): ##Post-activation quantizer scale
                max_range = 31* 0.029902491718530655                         #Pre-activation quantizer scale
                min_range = -31*0.029902491718530655                      #Pre-activation quantizer scale
                # max_range = round(max_range)
                # min_range = round(min_range)
                print("Tanh2",quant_scale)

            if(bit_width == 8 and max_range == 128):
                int_input_range = np.linspace(-1.002, 0.999, num=255)#Had to set this to -0.502 because the first input that the MT node got was -0.5019 which was less than -0.501 that I had set earlier.
            if(bit_width == 8 and max_range == 255):
                int_input_range = np.linspace(-0.502, 0.499, num=255)
            if(bit_width == 6):
                int_input_range = np.linspace(min_range,max_range,num=63) #63 thresholds for 6-bit quantization

                
            #int_input_range = np.linspace(-0.502, 0.499, num=255)#Had to set this to -0.502 because the first input that the MT node got was -0.5019 which was less than -0.501 that I had set earlier. Similariry had to set this from 0.498 to 0.499 as it got input 0.498012 which was greater than 0.498            int_input_range = torch.from_numpy(int_input_range) #conversion to torch tensor
            int_input_range = torch.from_numpy(int_input_range) #conversion to torch tensor
            tanh_out = nn.Tanh()              #defining sigmoid activation
            output = tanh_out(int_input_range)   #output of the activation function

            #QuantizeLinear operation
            # zero_point = 128 #This has to be 128 for Tanh with the range [-1,1] for 8-bit quantization
            zero_point = 0 #This has to be 0 for Tanh with 6-bit quantization
            #We need to implement this operation exactly according to the parameters speicified in the QCDQ graph.
            output = torch.round(output/quant_scale)
            #Factoring in the zero point
            output = output + zero_point 

            #Need to clip the outputs here to repliacte the fucnctioning of the Quant Node.
            #Clamping operation
            min_int_val = -31
            max_int_val = 31
            output = np.where(output > max_int_val,max_int_val, output)
            output = np.where(output < min_int_val,min_int_val, output)

            unique_input_indices = np.unique(output, return_index=True)[1] #[1:] #These are inputs which cause a change in the levels of the integer outputs from the quantization function
            unique_inputs = np.zeros(len(unique_input_indices))
            unique_inputs = int_input_range[unique_input_indices]#[1:] #identifying these inputs from the input_range array and ignoring the first threshold
            acivation_bit_width = bit_width
            num_thresholds = int(2 ** acivation_bit_width - 1)
            thresholds = np.zeros(num_thresholds)

            threshold_index = 0
            output_index = 0
            index_pos = 0
            first_value_zero_flag = 0
            if output[0] != 0:
                first_value_zero_flag = 1
            while threshold_index < 63 and output_index < 62 and index_pos < len(unique_input_indices):
                if output[output_index] != output[output_index+1]:
                    if(index_pos == 0):
                        diff = output[0]
                    elif(index_pos != 0):
                        diff = output[output_index+1] - output[output_index] # Calculating number of required repeats
                    while diff > 0:  #Copying repeats into the threshold thershold matrix       
                        thresholds[threshold_index] = unique_inputs[index_pos] 
                        threshold_index += 1
                        diff -= 1
                    index_pos += 1
                    if first_value_zero_flag == 1:
                        first_value_zero_flag = 0
                        continue
                    output_index += 1
                if output[output_index] == output[output_index+1]: 
                    output_index += 1
            
            #This for loop copies the last index of the unique index list to fill the remaining spaces in the thresholding matrix.
            for i in range(threshold_index, 63):
                if(index_pos != len(unique_input_indices)):
                    thresholds[i] = (int_input_range[unique_input_indices[index_pos]])
                else:
                    thresholds[i] = max_range+1

#Transformations Updated or new for the LSTM compute graph streamlining

class MoveScalarMulPastMatMul(Transformation):
    """Move scalar mul operations past matmul operations. We want to have muls
    next to each other such that they can be collapsed into a single mul."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Mul" and not model.is_fork_node(n) and not model.is_join_node(n):
                # print(n.input[1])
                if(n.output[0]=='global_out_1' or n.output[0]=='global_out_2'):
                    continue
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "MatMul"
                    and not model.is_join_node(consumer)
                ):
                    mul_weight_name = n.input[1]
                    matmul_weight_name = consumer.input[0]
                    print(mul_weight_name)
                    print(matmul_weight_name)
                    A = model.get_initializer(mul_weight_name)
                    W = model.get_initializer(matmul_weight_name)
                    if (A is None) or (W is None):
                        warnings.warn("MatMul or Mul params are not constant, skipping")
                        continue
                    # print("Here1")
                    start_name = n.input[0]
                    middle_name = n.output[0]
                    end_name = consumer.output[0]
                    mm_out_shape = model.get_tensor_shape(end_name)
                    if all(x == 1 for x in A.shape):
                        # if the mul is scalar, we can simply swap the order of ops
                        # make and insert new nodes
                        new_matmul = oh.make_node(
                            "MatMul",
                            [matmul_weight_name,start_name], #Shashwat Change
                            [middle_name],
                            name=consumer.name,
                        )
                        new_mul = oh.make_node(
                            "Mul",
                            [middle_name, mul_weight_name],
                            [end_name],
                            name=n.name,
                        )
                        #print("Here2")
                        graph.node.insert(node_ind, new_matmul)
                        graph.node.insert(node_ind + 1, new_mul)
                        model.set_tensor_shape(middle_name, mm_out_shape)
                        # remove old nodes
                        graph.node.remove(n)
                        graph.node.remove(consumer)
                        graph_modified = True
                        print("-----------------------------------------")
                        
            elif n.op_type == "Mul" and model.is_fork_node(n): #qlstm_change, only allowing fork nodes to pass through to the below  transformation
                #Shashwat change
                # print(n.input[1])
                break_flag = 0
                if (n.input[1] == "Mul_4_param0"):
                    continue
                consumer = model.find_consumer(n.output[0])
                for i in range(len(consumer)): #Loop for making sure all consumers are MatMul's
                    if(consumer[i].op_type != "MatMul"):
                        break_flag = 1
                        continue #This continue only continues the for loop not the big while loop
                if(break_flag == 1): #Hence this continue was introdcued so the transformation does not mess up with the Mul node present later in the graph.
                    continue
                for i in range(len(consumer)): #for loop for moving scalar mul past each consumer in the fork
                    if ( consumer[i] is not None and consumer[i].op_type == "MatMul" and not model.is_join_node(consumer[i])):
                        mul_weight_name = n.input[1]
                        matmul_weight_name = consumer[i].input[0] #qlstm_change [1] -> [0] : Matmul params are at index [0] due to shape constraints
                        A = model.get_initializer(mul_weight_name)
                        W = model.get_initializer(matmul_weight_name)
                        if (A is None) or (W is None):
                            warnings.warn("MatMul or Mul params are not constant, skipping")
                            continue
                        start_name = n.input[0]
                        middle_name = n.output[0]+str(i) #Update the middle name as it was common in all the four outputs. 
                        #So all four matmul's having an output to a single scalar mul node. Instead of one having output to all the mul's
                        end_name = consumer[i].output[0]
                        mm_out_shape = model.get_tensor_shape(end_name)
                        if all(x == 1 for x in A.shape):
                            # if the mul is scalar, we can simply swap the order of ops
                            # make and insert new nodes
                            new_matmul = oh.make_node(
                                "MatMul",
                                [matmul_weight_name,start_name], #Getting incompatible shapes that is why reversing the order of inputs in the node specification
                                [middle_name],
                                name=consumer[i].name,
                            )
                            new_mul = oh.make_node(
                                "Mul",
                                [middle_name, mul_weight_name],
                                [end_name],
                                name=n.name,
                            )
                            graph.node.insert(node_ind, new_matmul)
                            graph.node.insert(node_ind + 1, new_mul)
                            model.set_tensor_shape(middle_name, mm_out_shape)
                            # remove old nodes
                            graph.node.remove(consumer[i])
                graph.node.remove(n) #Removing the mul node after it has been moves past all consumers in the for loop
                graph_modified = True
        # model = model.transform(InferShapes())
        return (model, graph_modified)

# New transformation for moving the Scalar Mul's past the EltwiseMul
class MoveLinearPastEltwiseMul(Transformation):
    """Move linear operations (mul) past elementwise mul operations where possible.
    Specifically,matches and transforms the following patterns:
    (x*A) * (y*B) -> (xy)*(A*B)
    where x and y are dynamic inputs, A, B are constant tensors (in general).
    """

    def move_node(self, graph, n, prod0, prod1, node_ind):
        # found! move one of the muls to output, remove the other one
        lin0_in0 = prod0.input[0]
        lin1_in0 = prod1.input[0]
        in0 = n.input[0]
        out = n.output[0]
        # TODO: check shapes don't change through scalar mul or add
        # connect the eltwise add inputs to mul inputs
        n.input[0] = lin0_in0
        n.input[1] = lin1_in0
        # connect mul0 output to eltwise add output
        prod0.output[0] = out
        # connect the input of mul0 and output of eltwise add together
        n.output[0] = in0
        prod0.input[0] = in0
        # move prod0 node past eltwise add node, and remove prod1
        graph.node.remove(prod1)
        graph.node.remove(prod0)
        graph.node.insert(node_ind - 2, prod0)

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        nodes = [n for n in graph.node]
        for n in nodes:
            node_ind += 1
            if n.op_type == "Mul": #Checking if node is Eltwisemul.
                # scalar add has an initializer on one input
                in0 = n.input[0]
                in1 = n.input[1]
                if in0 is None or in1 is None:
                    continue
                A = model.get_initializer(in0)
                B = model.get_initializer(in1)
                if A is not None or B is not None:
                    continue
                # check for mul with same initializer on both inputs
                prod0 = model.find_producer(in0)
                prod1 = model.find_producer(in1)
                # Also check case when both branches are empty and come
                # from the same node: (prod0 == prod1)
                # Other transform should handle that
                if prod0 is None or prod1 is None or (prod0 == prod1):
                    continue
                if len(prod0.input) < 2 or len(prod1.input) < 2:
                    continue
                init0 = model.get_initializer(prod0.input[1])
                init1 = model.get_initializer(prod1.input[1])
                # if either initializer is None, skip
                if init0 is None or init1 is None:
                    continue
                if prod0.op_type == "Mul" and prod1.op_type == "Mul": #  (x*A) * (y*B) -> (xy)*(AB)
                    # Adding the update intializer condition in case of EltwiseMul.
                    init = init0*init1                         #Updating the initializer of the node which will move past the EltwiseMul
                    model.set_initializer(prod0.input[1],init) # update initializer of prod0, the node which will move.
                    self.move_node(graph,n,prod0,prod1,node_ind)
                    node_ind -= 1
                    graph_modified = True
                else:
                    continue
        model = model.transform(InferShapes())
        return (model, graph_modified)

class AbsorbMulIntoMultiThreshold(Transformation):
    """Absorb preceding Mul ops into MultiThreshold by updating the threshold
    values. Only *positive* scalar/1D mul vectors can be absorbed."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Mul" and not model.is_fork_node(n) and not model.is_join_node(n):
                mul_weight_name = n.input[1]
                print(mul_weight_name)
                # print(n.output[0])
                # This if condition is added so that the Mul node which gives the cell state as its output,
                # Does not get merged in the Multithreshold node, otherwise the output is hanging. 
                # TODO_lstm : Will have to generalize this further.........
                # if(n.output[0]=='global_out_1' or n.output[0]=='global_out_2'):
                    # continue
                A = model.get_initializer(mul_weight_name)
                print(A)
                # Shashwat Change : This 'if' statement helps in passing the 'Mul' node which has the global input cell_state
                # as it's input. So that all the remaining Mul operators can be absorbed into the Multithreshold node.
                # if (mul_weight_name == "Mul_10_param0" or mul_weight_name == "Mul_5_param0" or mul_weight_name == "Mul_6_param0" or mul_weight_name == "Mul_0_param0" or mul_weight_name == "Mul_4_param0" or mul_weight_name == "Mul_7_param0" or mul_weight_name == "Mul_9_param0" or mul_weight_name == "Mul_8_param0" or mul_weight_name == "Mul_11_param0" or mul_weight_name == "Mul_12_param0" or mul_weight_name == "Mul_15_param0" or mul_weight_name == "Mul_16_param0" or mul_weight_name == "Mul_19_param0" or mul_weight_name == "Mul_18_param0" or mul_weight_name == "Mul_24_param0" or mul_weight_name == "Mul_28_param0" or mul_weight_name == "Mul_31_param0"):         
                #     mul_weight_name = mul_weight_name
                if A == None:
                    continue
                # assert A is not None, "Initializer for mul weights is not set."
                # print("Mul Weight Name = ",mul_weight_name)
                is_signed = (A < 0).any()
                is_scalar = A.ndim == 0 or all(x == 1 for x in A.shape)
                actual_ndims = len(tuple(filter(lambda x: x > 1, A.shape)))
                is_1d = actual_ndims == 1
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "MultiThreshold":
                    if not is_signed and (is_1d or is_scalar):
                        threshold_name = consumer.input[1]
                        T = model.get_initializer(threshold_name)
                        assert T is not None, "Initializer for thresholds is not set."
                        start_name = n.input[0]
                        # compute new thresholds and set initializer
                        Tnew = T / A.reshape(-1, 1)
                        # TODO: need to handle negative A values correctly; produce
                        # mul sign mask and merge into preceding matmul?
                        model.set_initializer(threshold_name, Tnew)
                        # wire add input directly to MultiThreshold
                        consumer.input[0] = start_name
                        # remove the mul node
                        graph.node.remove(n)
                        graph_modified = True

            elif n.op_type == "Mul" and model.is_fork_node(n): #qlstm_change, only allowing fork nodes to pass through to the below  transformation
                mul_weight_name = n.input[1]
                break_flag = 0
                print(mul_weight_name)
                if(n.output[0]=='global_out_1' or n.output[0]=='global_out_2'):
                    continue
                A = model.get_initializer(mul_weight_name)
                print(A)
                if A == None:
                    continue
                is_signed = (A < 0).any()
                is_scalar = A.ndim == 0 or all(x == 1 for x in A.shape)
                actual_ndims = len(tuple(filter(lambda x: x > 1, A.shape)))
                is_1d = actual_ndims == 1
                consumer = model.find_consumer(n.output[0])
                for i in range(len(consumer)): #Loop for making sure all consumers are MultiThreshold's
                    if(consumer[i].op_type != "MultiThreshold"):
                        break_flag = 1
                        continue
                if(break_flag == 1):
                    continue
                for i in range(len(consumer)): #for loop for absorbing scalar mul in each consumer MultiThreshold in the fork
                    if consumer[i] is not None and consumer[i].op_type == "MultiThreshold":
                        if not is_signed and (is_1d or is_scalar):
                            threshold_name = consumer[i].input[1]
                            T = model.get_initializer(threshold_name)
                            assert T is not None, "Initializer for thresholds is not set."
                            start_name = n.input[0]
                            # compute new thresholds and set initializer
                            Tnew = T / A.reshape(-1, 1)
                            # TODO: need to handle negative A values correctly; produce
                            # mul sign mask and merge into preceding matmul?
                            model.set_initializer(threshold_name, Tnew)
                            # wire add input directly to MultiThreshold
                            consumer[i].input[0] = start_name
                            # remove the mul node
                graph.node.remove(n)#Removing the mul node after it has been absorbed by all the consumer MT nodes in the for loop
                graph_modified = True

        return (model, graph_modified)


class AbsorbSignBiasIntoMultiThreshold(Transformation):
    """Absorb scalar bias originating from signed int export back into
    MultiThreshold and re-evaluate the output datatype."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            # search for (MultiThreshold, Add) pair
            node_ind += 1
            if (
                n.op_type == "MultiThreshold"
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Add":
                    mt_node = n
                    add_node = consumer
                    threshold_name = mt_node.input[1]
                    add_weight_name = add_node.input[1]
                    print("Threshold name : ",threshold_name)
                    print("Add Weight name : ",add_weight_name)
                    T = model.get_initializer(threshold_name)
                    A = model.get_initializer(add_weight_name)
                    # mt_inst_unsigned = getCustomOp(mt_node)
                    # usigned_dtype = mt_inst_unsigned.get_nodeattr("out_dtype")
                    # if(usigned_dtype == 'UINT8'): #Assuming that with UINT8 inputs the bias factor needs to be 0 for the MT node.
                    #     A=A
                    if (A is None) or (T is None):
                        warnings.warn("Threshold or add bias not constant, skipping")
                        continue
                    end_name = add_node.output[0]
                    # we can only absorb scalar adds
                    is_scalar = A.ndim == 0 or all(x == 1 for x in A.shape)
                    if not is_scalar:
                        continue
                    bias = A.flatten()[0]
                    # set MultiThreshold bias property
                    mt_inst = getCustomOp(mt_node)
                    bias += mt_inst.get_nodeattr("out_bias")
                    mt_inst.set_nodeattr("out_bias", bias)
                    graph_modified = True
                    # compute new DataType for MultiThreshold output
                    steps = T.shape[-1]
                    new_min = bias
                    new_max = steps + bias
                    odt = DataType.get_smallest_possible(steps).name.replace("UINT", "INT")
                    print(odt)
                    print("-----------------------------------")
                    odt = DataType[odt]
                    # assert odt.allowed(new_max) and odt.allowed(
                    #     new_min
                    # ), """Could
                    # not compute new MultiThreshold DataType (min = %d max = %d)""" % (
                    #     new_min,
                    #     new_max,
                    # )
                    # mt_inst.set_nodeattr("out_dtype", odt.name)
                    # print(odt)
                    # print(odt)
                    #-----------------------
                    # if(usigned_dtype != 'UINT8'):
                    #     assert odt.allowed(new_max) and odt.allowed(
                    #     new_min
                    #     ), """Could
                    #     not compute new MultiThreshold DataType (min = %d max = %d)""" % (
                    #         new_min,
                    #         new_max,
                    #     )
                    #     mt_inst.set_nodeattr("out_dtype", odt.name)
                    # if(usigned_dtype == 'UINT8'):
                    #     odt = 'UINT8'
                    #     usigned_dtype = ' '
                    #     mt_inst.set_nodeattr("out_dtype", 'UINT8')
                    #-------------------------
                    # print(odt.name)
                    #--------------------------------------------------------
                    if(threshold_name != "MultiThreshold_8_param0" and threshold_name != "MultiThreshold_15_param0"):
                        assert odt.allowed(new_max) and odt.allowed(
                        new_min
                        ), """Could
                        not compute new MultiThreshold DataType (min = %d max = %d)""" % (
                        new_min,
                        new_max,
                        )
                        mt_inst.set_nodeattr("out_dtype", odt.name)
                    if(threshold_name == "MultiThreshold_8_param0" or threshold_name == "MultiThreshold_15_param0"):
                        mt_inst.set_nodeattr("out_dtype", "INT7")
                    # remove Add node, rewire MultiThreshold
                    graph.node.remove(add_node)
                    mt_node.output[0] = end_name
                    # set datatype
                    model.set_tensor_datatype(end_name, odt)
        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

class MoveLinearPastEltwiseAdd(Transformation):
    """Move linear operations (mul, add) past elementwise add operations where possible.
    Specifically,matches and transforms the following patterns:
    (x*C) + (y*C) -> (x + y) * C
    (x+A) + (y+B) -> (x + y) + (A + B)
    where x and y are dynamic inputs, A, B, C are constant tensors (in general).
    """

    def move_node(self, graph, n, prod0, prod1, node_ind):
        # found! move one of the muls to output, remove the other one
        lin0_in0 = prod0.input[0]
        lin1_in0 = prod1.input[0]
        in0 = n.input[0]
        out = n.output[0]
        # TODO: check shapes don't change through scalar mul or add
        # connect the eltwise add inputs to mul inputs
        n.input[0] = lin0_in0
        n.input[1] = lin1_in0
        # connect mul0 output to eltwise add output
        prod0.output[0] = out
        # connect the input of mul0 and output of eltwise add together
        n.output[0] = in0
        prod0.input[0] = in0
        # move prod0 node past eltwise add node, and remove prod1
        graph.node.remove(prod1)
        graph.node.remove(prod0)
        graph.node.insert(node_ind - 2, prod0)

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        nodes = [n for n in graph.node]
        for n in nodes:
            node_ind += 1
            if n.op_type == "Add":
                in0 = n.input[0]
                in1 = n.input[1]
                if in0 is None or in1 is None:
                    continue
                A = model.get_initializer(in0)
                B = model.get_initializer(in1)
                if A is not None or B is not None:
                    continue
                # check for mul with same initializer on both inputs
                prod0 = model.find_producer(in0)
                prod1 = model.find_producer(in1)
                # Also check case when both branches are empty and come
                # from the same node: (prod0 == prod1)
                # Other transform should handle that
                #print("Move Mul past add - 1")
                if prod0 is None or prod1 is None: #or (prod0 == prod1)
                    continue
                if len(prod0.input) < 2 or len(prod1.input) < 2:
                    continue
                init0 = model.get_initializer(prod0.input[1])
                init1 = model.get_initializer(prod1.input[1])
                #print("Move Mul past add - 2")
                # print(prod0.op_type)
                # print(prod1.op_type)
                print("----------------------------")
                # if either initializer is None, skip
                if init0 is None or init1 is None:
                    continue
                if prod0.op_type == "Mul" and prod1.op_type == "Mul":
                    # print("here-1")
                    if np.array_equal(init0, init1): #Can be generalized further
                        # init = init0
                        # print("here-2")
                        # model.set_initializer(prod0.input[1], init)
                        self.move_node(graph, n, prod0, prod1, node_ind)
                        node_ind -= 1
                        graph_modified = True
                elif prod0.op_type == "Add" and prod1.op_type == "Add":
                    init = init0 + init1
                    # update initializer of prod0, which we'll move
                    model.set_initializer(prod0.input[1], init)
                    self.move_node(graph, n, prod0, prod1, node_ind)
                    node_ind -= 1
                    graph_modified = True
                else:
                    continue
        model = model.transform(InferShapes())
        return (model, graph_modified)



#This function needs to be updated in the absorb.py file

class AbsorbSignBiasIntoMultiThreshold(Transformation):
    """Absorb scalar bias originating from signed int export back into
    MultiThreshold and re-evaluate the output datatype."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            # search for (MultiThreshold, Add) pair
            node_ind += 1
            if (
                n.op_type == "MultiThreshold"
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Add":
                    mt_node = n
                    add_node = consumer
                    threshold_name = mt_node.input[1]
                    add_weight_name = add_node.input[1]
                    print("Threshold name : ",threshold_name)
                    print("Add Weight name : ",add_weight_name)
                    T = model.get_initializer(threshold_name)
                    A = model.get_initializer(add_weight_name)
                    # mt_inst_unsigned = getCustomOp(mt_node)
                    # usigned_dtype = mt_inst_unsigned.get_nodeattr("out_dtype")
                    # if(usigned_dtype == 'UINT8'): #Assuming that with UINT8 inputs the bias factor needs to be 0 for the MT node.
                    #     A=A
                    if (A is None) or (T is None):
                        warnings.warn("Threshold or add bias not constant, skipping")
                        continue
                    end_name = add_node.output[0]
                    # we can only absorb scalar adds
                    is_scalar = A.ndim == 0 or all(x == 1 for x in A.shape)
                    if not is_scalar:
                        continue
                    bias = A.flatten()[0]
                    # set MultiThreshold bias property
                    mt_inst = getCustomOp(mt_node)
                    bias += mt_inst.get_nodeattr("out_bias")
                    mt_inst.set_nodeattr("out_bias", bias)
                    graph_modified = True
                    # compute new DataType for MultiThreshold output
                    steps = T.shape[-1]
                    new_min = bias
                    new_max = steps + bias
                    odt = DataType.get_smallest_possible(steps).name.replace("UINT", "INT")
                    print(odt)
                    print("-----------------------------------")
                    odt = DataType[odt]
                    # assert odt.allowed(new_max) and odt.allowed(
                    #     new_min
                    # ), """Could
                    # not compute new MultiThreshold DataType (min = %d max = %d)""" % (
                    #     new_min,
                    #     new_max,
                    # )
                    # mt_inst.set_nodeattr("out_dtype", odt.name)
                    # print(odt)
                    # print(odt)
                    #-----------------------
                    # if(usigned_dtype != 'UINT8'):
                    #     assert odt.allowed(new_max) and odt.allowed(
                    #     new_min
                    #     ), """Could
                    #     not compute new MultiThreshold DataType (min = %d max = %d)""" % (
                    #         new_min,
                    #         new_max,
                    #     )
                    #     mt_inst.set_nodeattr("out_dtype", odt.name)
                    # if(usigned_dtype == 'UINT8'):
                    #     odt = 'UINT8'
                    #     usigned_dtype = ' '
                    #     mt_inst.set_nodeattr("out_dtype", 'UINT8')
                    #-------------------------
                    # print(odt.name)
                    #--------------------------------------------------------
                    if(threshold_name != "MultiThreshold_8_param0" and threshold_name != "MultiThreshold_15_param0"):
                        assert odt.allowed(new_max) and odt.allowed(
                        new_min
                        ), """Could
                        not compute new MultiThreshold DataType (min = %d max = %d)""" % (
                        new_min,
                        new_max,
                        )
                        mt_inst.set_nodeattr("out_dtype", odt.name)
                    if(threshold_name == "MultiThreshold_8_param0" or threshold_name == "MultiThreshold_15_param0"):
                        mt_inst.set_nodeattr("out_dtype", "INT7")
                    # remove Add node, rewire MultiThreshold
                    graph.node.remove(add_node)
                    mt_node.output[0] = end_name
                    # set datatype
                    model.set_tensor_datatype(end_name, odt)
        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

class FactorOutMulSignMagnitude(Transformation):
    """Split multiply-by-constant nodes into two multiply-by-constant nodes,
    where the first node is a bipolar vector (of signs) and the second is a
    vector of magnitudes."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Mul":
                mul_weight_name = n.input[1]
                print('Current Transformation : ',mul_weight_name)
                A = model.get_initializer(mul_weight_name)
                print(A)
                # assert A is not None, "Initializer for mul weights is not set."
                # Shashwat change : Skipping unsupported node type
                if (mul_weight_name == "Mul_24_param0" or mul_weight_name == "Mul_28_param0" or mul_weight_name == "Mul_31_param0"):         
                    mul_weight_name = mul_weight_name
                elif A == None:
                    continue
                is_scalar = np.prod(A.shape) == 1
                actual_ndims = len(tuple(filter(lambda x: x > 1, A.shape)))
                is_1d = actual_ndims == 1
                is_not_bipolar = model.get_tensor_datatype(mul_weight_name) != DataType["BIPOLAR"]
                is_signed = (A < 0).any()
                if is_signed and (is_scalar or is_1d) and is_not_bipolar:
                    start_name = n.input[0]
                    in_shape = model.get_tensor_shape(start_name)
                    middle_name = model.make_new_valueinfo_name()
                    model.set_tensor_shape(middle_name, in_shape)
                    sign_mul_param_name = model.make_new_valueinfo_name()
                    # create new mul node with sign(A) as the operand
                    sgn = np.sign(A)
                    model.set_initializer(sign_mul_param_name, sgn)
                    model.set_tensor_datatype(sign_mul_param_name, DataType["BIPOLAR"])
                    # replace original mul weight by magnitudes
                    model.set_initializer(mul_weight_name, np.abs(A))
                    new_mul = oh.make_node("Mul", [start_name, sign_mul_param_name], [middle_name])
                    n.input[0] = middle_name
                    graph.node.insert(node_ind - 1, new_mul)
                    graph_modified = True
        return (model, graph_modified)