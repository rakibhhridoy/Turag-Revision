А
Џџ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
А
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintџџџџџџџџџ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8дс
Ќ
*Adam/simple_rnn_1/simple_rnn_cell_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*;
shared_name,*Adam/simple_rnn_1/simple_rnn_cell_1/bias/v
Ѕ
>Adam/simple_rnn_1/simple_rnn_cell_1/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_1/simple_rnn_cell_1/bias/v*
_output_shapes
:2*
dtype0
Ш
6Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*G
shared_name86Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/v
С
JAdam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/v*
_output_shapes

:22*
dtype0
Д
,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*=
shared_name.,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/v
­
@Adam/simple_rnn_1/simple_rnn_cell_1/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/v*
_output_shapes

:2*
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:*
dtype0

Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*&
shared_nameAdam/dense_8/kernel/v

)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes

:2*
dtype0
Ќ
*Adam/simple_rnn_1/simple_rnn_cell_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*;
shared_name,*Adam/simple_rnn_1/simple_rnn_cell_1/bias/m
Ѕ
>Adam/simple_rnn_1/simple_rnn_cell_1/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_1/simple_rnn_cell_1/bias/m*
_output_shapes
:2*
dtype0
Ш
6Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*G
shared_name86Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/m
С
JAdam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/m*
_output_shapes

:22*
dtype0
Д
,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*=
shared_name.,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/m
­
@Adam/simple_rnn_1/simple_rnn_cell_1/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/m*
_output_shapes

:2*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:*
dtype0

Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*&
shared_nameAdam/dense_8/kernel/m

)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes

:2*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

#simple_rnn_1/simple_rnn_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#simple_rnn_1/simple_rnn_cell_1/bias

7simple_rnn_1/simple_rnn_cell_1/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_1/simple_rnn_cell_1/bias*
_output_shapes
:2*
dtype0
К
/simple_rnn_1/simple_rnn_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*@
shared_name1/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel
Г
Csimple_rnn_1/simple_rnn_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel*
_output_shapes

:22*
dtype0
І
%simple_rnn_1/simple_rnn_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*6
shared_name'%simple_rnn_1/simple_rnn_cell_1/kernel

9simple_rnn_1/simple_rnn_cell_1/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_1/simple_rnn_cell_1/kernel*
_output_shapes

:2*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:2*
dtype0

"serving_default_simple_rnn_1_inputPlaceholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ
т
StatefulPartitionedCallStatefulPartitionedCall"serving_default_simple_rnn_1_input%simple_rnn_1/simple_rnn_cell_1/kernel#simple_rnn_1/simple_rnn_cell_1/bias/simple_rnn_1/simple_rnn_cell_1/recurrent_kerneldense_8/kerneldense_8/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_856614

NoOpNoOp
.
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ш-
valueО-BЛ- BД-

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures*
Њ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec*
І
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
'
0
1
2
3
4*
'
0
1
2
3
4*
* 
А
non_trainable_variables

 layers
!metrics
"layer_regularization_losses
#layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
$trace_0
%trace_1
&trace_2
'trace_3* 
6
(trace_0
)trace_1
*trace_2
+trace_3* 
* 

,iter

-beta_1

.beta_2
	/decay
0learning_ratemambmcmdmevfvgvhvivj*

1serving_default* 

0
1
2*

0
1
2*
* 


2states
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
8trace_0
9trace_1
:trace_2
;trace_3* 
6
<trace_0
=trace_1
>trace_2
?trace_3* 
г
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
F_random_generator

kernel
recurrent_kernel
bias*
* 

0
1*

0
1*
* 

Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ltrace_0* 

Mtrace_0* 
^X
VARIABLE_VALUEdense_8/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_8/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_1/simple_rnn_cell_1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#simple_rnn_1/simple_rnn_cell_1/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

N0
O1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1
2*

0
1
2*
* 

Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

Utrace_0
Vtrace_1* 

Wtrace_0
Xtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
8
Y	variables
Z	keras_api
	[total
	\count*
8
]	variables
^	keras_api
	_total
	`count*
* 
* 
* 
* 
* 
* 
* 
* 
* 

[0
\1*

Y	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

_0
`1*

]	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE6Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/simple_rnn_1/simple_rnn_cell_1/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE6Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/simple_rnn_1/simple_rnn_cell_1/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp9simple_rnn_1/simple_rnn_cell_1/kernel/Read/ReadVariableOpCsimple_rnn_1/simple_rnn_cell_1/recurrent_kernel/Read/ReadVariableOp7simple_rnn_1/simple_rnn_cell_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp@Adam/simple_rnn_1/simple_rnn_cell_1/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_1/simple_rnn_cell_1/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp@Adam/simple_rnn_1/simple_rnn_cell_1/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_1/simple_rnn_cell_1/bias/v/Read/ReadVariableOpConst*%
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_857536
Њ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/bias%simple_rnn_1/simple_rnn_cell_1/kernel/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel#simple_rnn_1/simple_rnn_cell_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_8/kernel/mAdam/dense_8/bias/m,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/m6Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/m*Adam/simple_rnn_1/simple_rnn_cell_1/bias/mAdam/dense_8/kernel/vAdam/dense_8/bias/v,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/v6Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/v*Adam/simple_rnn_1/simple_rnn_cell_1/bias/v*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_857618Њш
R
Ш
H__inference_sequential_4_layer_call_and_return_conditional_losses_856876

inputsO
=simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resource:2L
>simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resource:2Q
?simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource:228
&dense_8_matmul_readvariableop_resource:25
'dense_8_biasadd_readvariableop_resource:
identityЂdense_8/BiasAdd/ReadVariableOpЂdense_8/MatMul/ReadVariableOpЂ5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOpЂ6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOpЂsimple_rnn_1/whileH
simple_rnn_1/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_1/strided_sliceStridedSlicesimple_rnn_1/Shape:output:0)simple_rnn_1/strided_slice/stack:output:0+simple_rnn_1/strided_slice/stack_1:output:0+simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_1/zeros/packedPack#simple_rnn_1/strided_slice:output:0$simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
simple_rnn_1/zerosFill"simple_rnn_1/zeros/packed:output:0!simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2p
simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
simple_rnn_1/transpose	Transposeinputs$simple_rnn_1/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ^
simple_rnn_1/Shape_1Shapesimple_rnn_1/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_1/strided_slice_1StridedSlicesimple_rnn_1/Shape_1:output:0+simple_rnn_1/strided_slice_1/stack:output:0-simple_rnn_1/strided_slice_1/stack_1:output:0-simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџл
simple_rnn_1/TensorArrayV2TensorListReserve1simple_rnn_1/TensorArrayV2/element_shape:output:0%simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Bsimple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
4simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_1/transpose:y:0Ksimple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвl
"simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
simple_rnn_1/strided_slice_2StridedSlicesimple_rnn_1/transpose:y:0+simple_rnn_1/strided_slice_2/stack:output:0-simple_rnn_1/strided_slice_2/stack_1:output:0-simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskВ
4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp=simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0Ц
%simple_rnn_1/simple_rnn_cell_1/MatMulMatMul%simple_rnn_1/strided_slice_2:output:0<simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2А
5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0г
&simple_rnn_1/simple_rnn_cell_1/BiasAddBiasAdd/simple_rnn_1/simple_rnn_cell_1/MatMul:product:0=simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ж
6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype0Р
'simple_rnn_1/simple_rnn_cell_1/MatMul_1MatMulsimple_rnn_1/zeros:output:0>simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2С
"simple_rnn_1/simple_rnn_cell_1/addAddV2/simple_rnn_1/simple_rnn_cell_1/BiasAdd:output:01simple_rnn_1/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
#simple_rnn_1/simple_rnn_cell_1/ReluRelu&simple_rnn_1/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2{
*simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   k
)simple_rnn_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ь
simple_rnn_1/TensorArrayV2_1TensorListReserve3simple_rnn_1/TensorArrayV2_1/element_shape:output:02simple_rnn_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвS
simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџa
simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
simple_rnn_1/whileWhile(simple_rnn_1/while/loop_counter:output:0.simple_rnn_1/while/maximum_iterations:output:0simple_rnn_1/time:output:0%simple_rnn_1/TensorArrayV2_1:handle:0simple_rnn_1/zeros:output:0%simple_rnn_1/strided_slice_1:output:0Dsimple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resource>simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resource?simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( **
body"R 
simple_rnn_1_while_body_856803**
cond"R 
simple_rnn_1_while_cond_856802*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 
=simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   §
/simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_1/while:output:3Fsimple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ2*
element_dtype0*
num_elementsu
"simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџn
$simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ш
simple_rnn_1/strided_slice_3StridedSlice8simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_1/strided_slice_3/stack:output:0-simple_rnn_1/strided_slice_3/stack_1:output:0-simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maskr
simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
simple_rnn_1/transpose_1	Transpose8simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
dense_8/MatMulMatMul%simple_rnn_1/strided_slice_3:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџg
IdentityIdentitydense_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџФ
NoOpNoOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp6^simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp5^simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp7^simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp^simple_rnn_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2n
5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp2l
4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp2p
6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp2(
simple_rnn_1/whilesimple_rnn_1/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
"
б
while_body_856129
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
 while_simple_rnn_cell_1_856151_0:2.
 while_simple_rnn_cell_1_856153_0:22
 while_simple_rnn_cell_1_856155_0:22
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_1_856151:2,
while_simple_rnn_cell_1_856153:20
while_simple_rnn_cell_1_856155:22Ђ/while/simple_rnn_cell_1/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ђ
/while/simple_rnn_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2 while_simple_rnn_cell_1_856151_0 while_simple_rnn_cell_1_856153_0 while_simple_rnn_cell_1_856155_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_856076r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:08while/simple_rnn_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity8while/simple_rnn_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2~

while/NoOpNoOp0^while/simple_rnn_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"B
while_simple_rnn_cell_1_856151 while_simple_rnn_cell_1_856151_0"B
while_simple_rnn_cell_1_856153 while_simple_rnn_cell_1_856153_0"B
while_simple_rnn_cell_1_856155 while_simple_rnn_cell_1_856155_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ2: : : : : 2b
/while/simple_rnn_cell_1/StatefulPartitionedCall/while/simple_rnn_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
ј4

H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_856193

inputs*
simple_rnn_cell_1_856116:2&
simple_rnn_cell_1_856118:2*
simple_rnn_cell_1_856120:22
identityЂ)simple_rnn_cell_1/StatefulPartitionedCallЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskч
)simple_rnn_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_1_856116simple_rnn_cell_1_856118simple_rnn_cell_1_856120*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_856076n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_1_856116simple_rnn_cell_1_856118simple_rnn_cell_1_856120*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_856129*
condR
while_cond_856128*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ2*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2z
NoOpNoOp*^simple_rnn_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2V
)simple_rnn_cell_1/StatefulPartitionedCall)simple_rnn_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
К

Ё
simple_rnn_1_while_cond_8568026
2simple_rnn_1_while_simple_rnn_1_while_loop_counter<
8simple_rnn_1_while_simple_rnn_1_while_maximum_iterations"
simple_rnn_1_while_placeholder$
 simple_rnn_1_while_placeholder_1$
 simple_rnn_1_while_placeholder_28
4simple_rnn_1_while_less_simple_rnn_1_strided_slice_1N
Jsimple_rnn_1_while_simple_rnn_1_while_cond_856802___redundant_placeholder0N
Jsimple_rnn_1_while_simple_rnn_1_while_cond_856802___redundant_placeholder1N
Jsimple_rnn_1_while_simple_rnn_1_while_cond_856802___redundant_placeholder2N
Jsimple_rnn_1_while_simple_rnn_1_while_cond_856802___redundant_placeholder3
simple_rnn_1_while_identity

simple_rnn_1/while/LessLesssimple_rnn_1_while_placeholder4simple_rnn_1_while_less_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_1/while/IdentityIdentitysimple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_1_while_identity$simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
в
э
-__inference_sequential_4_layer_call_fn_856644

inputs
unknown:2
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_856531o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
с`
М
!__inference__wrapped_model_855906
simple_rnn_1_input\
Jsequential_4_simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resource:2Y
Ksequential_4_simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resource:2^
Lsequential_4_simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource:22E
3sequential_4_dense_8_matmul_readvariableop_resource:2B
4sequential_4_dense_8_biasadd_readvariableop_resource:
identityЂ+sequential_4/dense_8/BiasAdd/ReadVariableOpЂ*sequential_4/dense_8/MatMul/ReadVariableOpЂBsequential_4/simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂAsequential_4/simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOpЂCsequential_4/simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOpЂsequential_4/simple_rnn_1/whilea
sequential_4/simple_rnn_1/ShapeShapesimple_rnn_1_input*
T0*
_output_shapes
:w
-sequential_4/simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/sequential_4/simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/sequential_4/simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
'sequential_4/simple_rnn_1/strided_sliceStridedSlice(sequential_4/simple_rnn_1/Shape:output:06sequential_4/simple_rnn_1/strided_slice/stack:output:08sequential_4/simple_rnn_1/strided_slice/stack_1:output:08sequential_4/simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_4/simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2С
&sequential_4/simple_rnn_1/zeros/packedPack0sequential_4/simple_rnn_1/strided_slice:output:01sequential_4/simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%sequential_4/simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    К
sequential_4/simple_rnn_1/zerosFill/sequential_4/simple_rnn_1/zeros/packed:output:0.sequential_4/simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2}
(sequential_4/simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ­
#sequential_4/simple_rnn_1/transpose	Transposesimple_rnn_1_input1sequential_4/simple_rnn_1/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџx
!sequential_4/simple_rnn_1/Shape_1Shape'sequential_4/simple_rnn_1/transpose:y:0*
T0*
_output_shapes
:y
/sequential_4/simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_4/simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_4/simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)sequential_4/simple_rnn_1/strided_slice_1StridedSlice*sequential_4/simple_rnn_1/Shape_1:output:08sequential_4/simple_rnn_1/strided_slice_1/stack:output:0:sequential_4/simple_rnn_1/strided_slice_1/stack_1:output:0:sequential_4/simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5sequential_4/simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
'sequential_4/simple_rnn_1/TensorArrayV2TensorListReserve>sequential_4/simple_rnn_1/TensorArrayV2/element_shape:output:02sequential_4/simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв 
Osequential_4/simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ў
Asequential_4/simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_4/simple_rnn_1/transpose:y:0Xsequential_4/simple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвy
/sequential_4/simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_4/simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_4/simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
)sequential_4/simple_rnn_1/strided_slice_2StridedSlice'sequential_4/simple_rnn_1/transpose:y:08sequential_4/simple_rnn_1/strided_slice_2/stack:output:0:sequential_4/simple_rnn_1/strided_slice_2/stack_1:output:0:sequential_4/simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskЬ
Asequential_4/simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpJsequential_4_simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0э
2sequential_4/simple_rnn_1/simple_rnn_cell_1/MatMulMatMul2sequential_4/simple_rnn_1/strided_slice_2:output:0Isequential_4/simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ъ
Bsequential_4/simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpKsequential_4_simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0њ
3sequential_4/simple_rnn_1/simple_rnn_cell_1/BiasAddBiasAdd<sequential_4/simple_rnn_1/simple_rnn_cell_1/MatMul:product:0Jsequential_4/simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2а
Csequential_4/simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpLsequential_4_simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype0ч
4sequential_4/simple_rnn_1/simple_rnn_cell_1/MatMul_1MatMul(sequential_4/simple_rnn_1/zeros:output:0Ksequential_4/simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2ш
/sequential_4/simple_rnn_1/simple_rnn_cell_1/addAddV2<sequential_4/simple_rnn_1/simple_rnn_cell_1/BiasAdd:output:0>sequential_4/simple_rnn_1/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
0sequential_4/simple_rnn_1/simple_rnn_cell_1/ReluRelu3sequential_4/simple_rnn_1/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
7sequential_4/simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   x
6sequential_4/simple_rnn_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
)sequential_4/simple_rnn_1/TensorArrayV2_1TensorListReserve@sequential_4/simple_rnn_1/TensorArrayV2_1/element_shape:output:0?sequential_4/simple_rnn_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв`
sequential_4/simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : }
2sequential_4/simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџn
,sequential_4/simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Њ
sequential_4/simple_rnn_1/whileWhile5sequential_4/simple_rnn_1/while/loop_counter:output:0;sequential_4/simple_rnn_1/while/maximum_iterations:output:0'sequential_4/simple_rnn_1/time:output:02sequential_4/simple_rnn_1/TensorArrayV2_1:handle:0(sequential_4/simple_rnn_1/zeros:output:02sequential_4/simple_rnn_1/strided_slice_1:output:0Qsequential_4/simple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Jsequential_4_simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resourceKsequential_4_simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resourceLsequential_4_simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *7
body/R-
+sequential_4_simple_rnn_1_while_body_855833*7
cond/R-
+sequential_4_simple_rnn_1_while_cond_855832*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 
Jsequential_4/simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   Є
<sequential_4/simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_4/simple_rnn_1/while:output:3Ssequential_4/simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ2*
element_dtype0*
num_elements
/sequential_4/simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1sequential_4/simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1sequential_4/simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
)sequential_4/simple_rnn_1/strided_slice_3StridedSliceEsequential_4/simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:08sequential_4/simple_rnn_1/strided_slice_3/stack:output:0:sequential_4/simple_rnn_1/strided_slice_3/stack_1:output:0:sequential_4/simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask
*sequential_4/simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ф
%sequential_4/simple_rnn_1/transpose_1	TransposeEsequential_4/simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:03sequential_4/simple_rnn_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
*sequential_4/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_8_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0П
sequential_4/dense_8/MatMulMatMul2sequential_4/simple_rnn_1/strided_slice_3:output:02sequential_4/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
+sequential_4/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Е
sequential_4/dense_8/BiasAddBiasAdd%sequential_4/dense_8/MatMul:product:03sequential_4/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџt
IdentityIdentity%sequential_4/dense_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp,^sequential_4/dense_8/BiasAdd/ReadVariableOp+^sequential_4/dense_8/MatMul/ReadVariableOpC^sequential_4/simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpB^sequential_4/simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOpD^sequential_4/simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp ^sequential_4/simple_rnn_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : 2Z
+sequential_4/dense_8/BiasAdd/ReadVariableOp+sequential_4/dense_8/BiasAdd/ReadVariableOp2X
*sequential_4/dense_8/MatMul/ReadVariableOp*sequential_4/dense_8/MatMul/ReadVariableOp2
Bsequential_4/simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpBsequential_4/simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp2
Asequential_4/simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOpAsequential_4/simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp2
Csequential_4/simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOpCsequential_4/simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp2B
sequential_4/simple_rnn_1/whilesequential_4/simple_rnn_1/while:_ [
+
_output_shapes
:џџџџџџџџџ
,
_user_specified_namesimple_rnn_1_input
Љ
э
H__inference_sequential_4_layer_call_and_return_conditional_losses_856591
simple_rnn_1_input%
simple_rnn_1_856578:2!
simple_rnn_1_856580:2%
simple_rnn_1_856582:22 
dense_8_856585:2
dense_8_856587:
identityЂdense_8/StatefulPartitionedCallЂ$simple_rnn_1/StatefulPartitionedCallЃ
$simple_rnn_1/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_1_inputsimple_rnn_1_856578simple_rnn_1_856580simple_rnn_1_856582*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_856489
dense_8/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_1/StatefulPartitionedCall:output:0dense_8_856585dense_8_856587*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_856336w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^dense_8/StatefulPartitionedCall%^simple_rnn_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2L
$simple_rnn_1/StatefulPartitionedCall$simple_rnn_1/StatefulPartitionedCall:_ [
+
_output_shapes
:џџџџџџџџџ
,
_user_specified_namesimple_rnn_1_input


+sequential_4_simple_rnn_1_while_cond_855832P
Lsequential_4_simple_rnn_1_while_sequential_4_simple_rnn_1_while_loop_counterV
Rsequential_4_simple_rnn_1_while_sequential_4_simple_rnn_1_while_maximum_iterations/
+sequential_4_simple_rnn_1_while_placeholder1
-sequential_4_simple_rnn_1_while_placeholder_11
-sequential_4_simple_rnn_1_while_placeholder_2R
Nsequential_4_simple_rnn_1_while_less_sequential_4_simple_rnn_1_strided_slice_1h
dsequential_4_simple_rnn_1_while_sequential_4_simple_rnn_1_while_cond_855832___redundant_placeholder0h
dsequential_4_simple_rnn_1_while_sequential_4_simple_rnn_1_while_cond_855832___redundant_placeholder1h
dsequential_4_simple_rnn_1_while_sequential_4_simple_rnn_1_while_cond_855832___redundant_placeholder2h
dsequential_4_simple_rnn_1_while_sequential_4_simple_rnn_1_while_cond_855832___redundant_placeholder3,
(sequential_4_simple_rnn_1_while_identity
Ъ
$sequential_4/simple_rnn_1/while/LessLess+sequential_4_simple_rnn_1_while_placeholderNsequential_4_simple_rnn_1_while_less_sequential_4_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: 
(sequential_4/simple_rnn_1/while/IdentityIdentity(sequential_4/simple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "]
(sequential_4_simple_rnn_1_while_identity1sequential_4/simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Ы>
Н
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_857030
inputs_0B
0simple_rnn_cell_1_matmul_readvariableop_resource:2?
1simple_rnn_cell_1_biasadd_readvariableop_resource:2D
2simple_rnn_cell_1_matmul_1_readvariableop_resource:22
identityЂ(simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_1/MatMul/ReadVariableOpЂ)simple_rnn_cell_1/MatMul_1/ReadVariableOpЂwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0Ќ
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype0
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2k
simple_rnn_cell_1/ReluRelusimple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_856963*
condR
while_cond_856962*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ2*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2Я
NoOpNoOp)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0

ш
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_855954

inputs

states0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:22
 matmul_1_readvariableop_resource:22
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2G
ReluReluadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates
Ј>
Л
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_857250

inputsB
0simple_rnn_cell_1_matmul_readvariableop_resource:2?
1simple_rnn_cell_1_biasadd_readvariableop_resource:2D
2simple_rnn_cell_1_matmul_1_readvariableop_resource:22
identityЂ(simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_1/MatMul/ReadVariableOpЂ)simple_rnn_cell_1/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0Ќ
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype0
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2k
simple_rnn_cell_1/ReluRelusimple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_857183*
condR
while_cond_857182*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ2*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2Я
NoOpNoOp)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ш
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_856076

inputs

states0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:22
 matmul_1_readvariableop_resource:22
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2G
ReluReluadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates

Й
-__inference_simple_rnn_1_layer_call_fn_856898
inputs_0
unknown:2
	unknown_0:2
	unknown_1:22
identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_856193o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
л-
Ш
while_body_857293
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0:2G
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:2L
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:22
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_1_matmul_readvariableop_resource:2E
7while_simple_rnn_cell_1_biasadd_readvariableop_resource:2J
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:22Ђ.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_1/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0І
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0У
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Є
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype0О
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Њ
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype0Њ
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ќ
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2w
while/simple_rnn_cell_1/ReluReluwhile/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ћ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_1/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity*while/simple_rnn_cell_1/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2п

while/NoOpNoOp/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ2: : : : : 2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Р

(__inference_dense_8_layer_call_fn_857369

inputs
unknown:2
	unknown_0:
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_856336o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Ј>
Л
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_857360

inputsB
0simple_rnn_cell_1_matmul_readvariableop_resource:2?
1simple_rnn_cell_1_biasadd_readvariableop_resource:2D
2simple_rnn_cell_1_matmul_1_readvariableop_resource:22
identityЂ(simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_1/MatMul/ReadVariableOpЂ)simple_rnn_cell_1/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0Ќ
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype0
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2k
simple_rnn_cell_1/ReluRelusimple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_857293*
condR
while_cond_857292*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ2*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2Я
NoOpNoOp)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к
Њ
while_cond_856421
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_856421___redundant_placeholder04
0while_while_cond_856421___redundant_placeholder14
0while_while_cond_856421___redundant_placeholder24
0while_while_cond_856421___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Њc
э
"__inference__traced_restore_857618
file_prefix1
assignvariableop_dense_8_kernel:2-
assignvariableop_1_dense_8_bias:J
8assignvariableop_2_simple_rnn_1_simple_rnn_cell_1_kernel:2T
Bassignvariableop_3_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel:22D
6assignvariableop_4_simple_rnn_1_simple_rnn_cell_1_bias:2&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: #
assignvariableop_12_total: #
assignvariableop_13_count: ;
)assignvariableop_14_adam_dense_8_kernel_m:25
'assignvariableop_15_adam_dense_8_bias_m:R
@assignvariableop_16_adam_simple_rnn_1_simple_rnn_cell_1_kernel_m:2\
Jassignvariableop_17_adam_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_m:22L
>assignvariableop_18_adam_simple_rnn_1_simple_rnn_cell_1_bias_m:2;
)assignvariableop_19_adam_dense_8_kernel_v:25
'assignvariableop_20_adam_dense_8_bias_v:R
@assignvariableop_21_adam_simple_rnn_1_simple_rnn_cell_1_kernel_v:2\
Jassignvariableop_22_adam_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_v:22L
>assignvariableop_23_adam_simple_rnn_1_simple_rnn_cell_1_bias_v:2
identity_25ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*К
valueАB­B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЂ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_2AssignVariableOp8assignvariableop_2_simple_rnn_1_simple_rnn_cell_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_3AssignVariableOpBassignvariableop_3_simple_rnn_1_simple_rnn_cell_1_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_4AssignVariableOp6assignvariableop_4_simple_rnn_1_simple_rnn_cell_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_dense_8_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_dense_8_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_16AssignVariableOp@assignvariableop_16_adam_simple_rnn_1_simple_rnn_cell_1_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_17AssignVariableOpJassignvariableop_17_adam_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_18AssignVariableOp>assignvariableop_18_adam_simple_rnn_1_simple_rnn_cell_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_8_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_8_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_simple_rnn_1_simple_rnn_cell_1_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_22AssignVariableOpJassignvariableop_22_adam_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_simple_rnn_1_simple_rnn_cell_1_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 п
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: Ь
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ў
З
-__inference_simple_rnn_1_layer_call_fn_856920

inputs
unknown:2
	unknown_0:2
	unknown_1:22
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_856489o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

с
H__inference_sequential_4_layer_call_and_return_conditional_losses_856343

inputs%
simple_rnn_1_856319:2!
simple_rnn_1_856321:2%
simple_rnn_1_856323:22 
dense_8_856337:2
dense_8_856339:
identityЂdense_8/StatefulPartitionedCallЂ$simple_rnn_1/StatefulPartitionedCall
$simple_rnn_1/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_1_856319simple_rnn_1_856321simple_rnn_1_856323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_856318
dense_8/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_1/StatefulPartitionedCall:output:0dense_8_856337dense_8_856339*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_856336w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^dense_8/StatefulPartitionedCall%^simple_rnn_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2L
$simple_rnn_1/StatefulPartitionedCall$simple_rnn_1/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ы>
Н
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_857140
inputs_0B
0simple_rnn_cell_1_matmul_readvariableop_resource:2?
1simple_rnn_cell_1_biasadd_readvariableop_resource:2D
2simple_rnn_cell_1_matmul_1_readvariableop_resource:22
identityЂ(simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_1/MatMul/ReadVariableOpЂ)simple_rnn_cell_1/MatMul_1/ReadVariableOpЂwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0Ќ
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype0
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2k
simple_rnn_cell_1/ReluRelusimple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_857073*
condR
while_cond_857072*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ2*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2Я
NoOpNoOp)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
к
Њ
while_cond_857072
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_857072___redundant_placeholder04
0while_while_cond_857072___redundant_placeholder14
0while_while_cond_857072___redundant_placeholder24
0while_while_cond_857072___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Й

к
2__inference_simple_rnn_cell_1_layer_call_fn_857393

inputs
states_0
unknown:2
	unknown_0:2
	unknown_1:22
identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_855954o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0
ў
З
-__inference_simple_rnn_1_layer_call_fn_856909

inputs
unknown:2
	unknown_0:2
	unknown_1:22
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_856318o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і
љ
-__inference_sequential_4_layer_call_fn_856356
simple_rnn_1_input
unknown:2
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_856343o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:џџџџџџџџџ
,
_user_specified_namesimple_rnn_1_input
ј4

H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_856032

inputs*
simple_rnn_cell_1_855955:2&
simple_rnn_cell_1_855957:2*
simple_rnn_cell_1_855959:22
identityЂ)simple_rnn_cell_1/StatefulPartitionedCallЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskч
)simple_rnn_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_1_855955simple_rnn_cell_1_855957simple_rnn_cell_1_855959*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_855954n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_1_855955simple_rnn_cell_1_855957simple_rnn_cell_1_855959*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_855968*
condR
while_cond_855967*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ2*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2z
NoOpNoOp*^simple_rnn_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2V
)simple_rnn_cell_1/StatefulPartitionedCall)simple_rnn_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
 F
и
+sequential_4_simple_rnn_1_while_body_855833P
Lsequential_4_simple_rnn_1_while_sequential_4_simple_rnn_1_while_loop_counterV
Rsequential_4_simple_rnn_1_while_sequential_4_simple_rnn_1_while_maximum_iterations/
+sequential_4_simple_rnn_1_while_placeholder1
-sequential_4_simple_rnn_1_while_placeholder_11
-sequential_4_simple_rnn_1_while_placeholder_2O
Ksequential_4_simple_rnn_1_while_sequential_4_simple_rnn_1_strided_slice_1_0
sequential_4_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0d
Rsequential_4_simple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0:2a
Ssequential_4_simple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:2f
Tsequential_4_simple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:22,
(sequential_4_simple_rnn_1_while_identity.
*sequential_4_simple_rnn_1_while_identity_1.
*sequential_4_simple_rnn_1_while_identity_2.
*sequential_4_simple_rnn_1_while_identity_3.
*sequential_4_simple_rnn_1_while_identity_4M
Isequential_4_simple_rnn_1_while_sequential_4_simple_rnn_1_strided_slice_1
sequential_4_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_1_tensorarrayunstack_tensorlistfromtensorb
Psequential_4_simple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource:2_
Qsequential_4_simple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource:2d
Rsequential_4_simple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource:22ЂHsequential_4/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂGsequential_4/simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpЂIsequential_4/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpЂ
Qsequential_4/simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Љ
Csequential_4/simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_4_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0+sequential_4_simple_rnn_1_while_placeholderZsequential_4/simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0к
Gsequential_4/simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpRsequential_4_simple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0
8sequential_4/simple_rnn_1/while/simple_rnn_cell_1/MatMulMatMulJsequential_4/simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Osequential_4/simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2и
Hsequential_4/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpSsequential_4_simple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype0
9sequential_4/simple_rnn_1/while/simple_rnn_cell_1/BiasAddBiasAddBsequential_4/simple_rnn_1/while/simple_rnn_cell_1/MatMul:product:0Psequential_4/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2о
Isequential_4/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpTsequential_4_simple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype0ј
:sequential_4/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1MatMul-sequential_4_simple_rnn_1_while_placeholder_2Qsequential_4/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2њ
5sequential_4/simple_rnn_1/while/simple_rnn_cell_1/addAddV2Bsequential_4/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd:output:0Dsequential_4/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ћ
6sequential_4/simple_rnn_1/while/simple_rnn_cell_1/ReluRelu9sequential_4/simple_rnn_1/while/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Jsequential_4/simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : у
Dsequential_4/simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_4_simple_rnn_1_while_placeholder_1Ssequential_4/simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem/index:output:0Dsequential_4/simple_rnn_1/while/simple_rnn_cell_1/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвg
%sequential_4/simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Њ
#sequential_4/simple_rnn_1/while/addAddV2+sequential_4_simple_rnn_1_while_placeholder.sequential_4/simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: i
'sequential_4/simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Я
%sequential_4/simple_rnn_1/while/add_1AddV2Lsequential_4_simple_rnn_1_while_sequential_4_simple_rnn_1_while_loop_counter0sequential_4/simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: Ї
(sequential_4/simple_rnn_1/while/IdentityIdentity)sequential_4/simple_rnn_1/while/add_1:z:0%^sequential_4/simple_rnn_1/while/NoOp*
T0*
_output_shapes
: в
*sequential_4/simple_rnn_1/while/Identity_1IdentityRsequential_4_simple_rnn_1_while_sequential_4_simple_rnn_1_while_maximum_iterations%^sequential_4/simple_rnn_1/while/NoOp*
T0*
_output_shapes
: Ї
*sequential_4/simple_rnn_1/while/Identity_2Identity'sequential_4/simple_rnn_1/while/add:z:0%^sequential_4/simple_rnn_1/while/NoOp*
T0*
_output_shapes
: д
*sequential_4/simple_rnn_1/while/Identity_3IdentityTsequential_4/simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^sequential_4/simple_rnn_1/while/NoOp*
T0*
_output_shapes
: е
*sequential_4/simple_rnn_1/while/Identity_4IdentityDsequential_4/simple_rnn_1/while/simple_rnn_cell_1/Relu:activations:0%^sequential_4/simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2Ч
$sequential_4/simple_rnn_1/while/NoOpNoOpI^sequential_4/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpH^sequential_4/simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpJ^sequential_4/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "]
(sequential_4_simple_rnn_1_while_identity1sequential_4/simple_rnn_1/while/Identity:output:0"a
*sequential_4_simple_rnn_1_while_identity_13sequential_4/simple_rnn_1/while/Identity_1:output:0"a
*sequential_4_simple_rnn_1_while_identity_23sequential_4/simple_rnn_1/while/Identity_2:output:0"a
*sequential_4_simple_rnn_1_while_identity_33sequential_4/simple_rnn_1/while/Identity_3:output:0"a
*sequential_4_simple_rnn_1_while_identity_43sequential_4/simple_rnn_1/while/Identity_4:output:0"
Isequential_4_simple_rnn_1_while_sequential_4_simple_rnn_1_strided_slice_1Ksequential_4_simple_rnn_1_while_sequential_4_simple_rnn_1_strided_slice_1_0"Ј
Qsequential_4_simple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resourceSsequential_4_simple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"Њ
Rsequential_4_simple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resourceTsequential_4_simple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"І
Psequential_4_simple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resourceRsequential_4_simple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0"
sequential_4_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_1_tensorarrayunstack_tensorlistfromtensorsequential_4_simple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ2: : : : : 2
Hsequential_4/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpHsequential_4/simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2
Gsequential_4/simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpGsequential_4/simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp2
Isequential_4/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpIsequential_4/simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 

ъ
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_857424

inputs
states_00
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:22
 matmul_1_readvariableop_resource:22
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2G
ReluReluadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0

с
H__inference_sequential_4_layer_call_and_return_conditional_losses_856531

inputs%
simple_rnn_1_856518:2!
simple_rnn_1_856520:2%
simple_rnn_1_856522:22 
dense_8_856525:2
dense_8_856527:
identityЂdense_8/StatefulPartitionedCallЂ$simple_rnn_1/StatefulPartitionedCall
$simple_rnn_1/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_1_856518simple_rnn_1_856520simple_rnn_1_856522*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_856489
dense_8/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_1/StatefulPartitionedCall:output:0dense_8_856525dense_8_856527*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_856336w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^dense_8/StatefulPartitionedCall%^simple_rnn_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2L
$simple_rnn_1/StatefulPartitionedCall$simple_rnn_1/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
л-
Ш
while_body_856422
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0:2G
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:2L
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:22
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_1_matmul_readvariableop_resource:2E
7while_simple_rnn_cell_1_biasadd_readvariableop_resource:2J
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:22Ђ.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_1/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0І
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0У
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Є
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype0О
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Њ
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype0Њ
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ќ
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2w
while/simple_rnn_cell_1/ReluReluwhile/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ћ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_1/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity*while/simple_rnn_cell_1/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2п

while/NoOpNoOp/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ2: : : : : 2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
л-
Ш
while_body_857183
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0:2G
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:2L
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:22
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_1_matmul_readvariableop_resource:2E
7while_simple_rnn_cell_1_biasadd_readvariableop_resource:2J
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:22Ђ.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_1/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0І
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0У
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Є
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype0О
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Њ
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype0Њ
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ќ
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2w
while/simple_rnn_cell_1/ReluReluwhile/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ћ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_1/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity*while/simple_rnn_cell_1/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2п

while/NoOpNoOp/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ2: : : : : 2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
л-
Ш
while_body_857073
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0:2G
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:2L
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:22
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_1_matmul_readvariableop_resource:2E
7while_simple_rnn_cell_1_biasadd_readvariableop_resource:2J
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:22Ђ.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_1/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0І
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0У
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Є
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype0О
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Њ
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype0Њ
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ќ
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2w
while/simple_rnn_cell_1/ReluReluwhile/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ћ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_1/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity*while/simple_rnn_cell_1/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2п

while/NoOpNoOp/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ2: : : : : 2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
к
Њ
while_cond_856250
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_856250___redundant_placeholder04
0while_while_cond_856250___redundant_placeholder14
0while_while_cond_856250___redundant_placeholder24
0while_while_cond_856250___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
9
а
__inference__traced_save_857536
file_prefix-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableopD
@savev2_simple_rnn_1_simple_rnn_cell_1_kernel_read_readvariableopN
Jsavev2_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_1_simple_rnn_cell_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_1_simple_rnn_cell_1_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_1_simple_rnn_cell_1_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_1_simple_rnn_cell_1_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_1_simple_rnn_cell_1_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*К
valueАB­B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B Ь
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop@savev2_simple_rnn_1_simple_rnn_cell_1_kernel_read_readvariableopJsavev2_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_read_readvariableop>savev2_simple_rnn_1_simple_rnn_cell_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableopGsavev2_adam_simple_rnn_1_simple_rnn_cell_1_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_1_simple_rnn_cell_1_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableopGsavev2_adam_simple_rnn_1_simple_rnn_cell_1_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_1_simple_rnn_cell_1_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_1_simple_rnn_cell_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Љ
_input_shapes
: :2::2:22:2: : : : : : : : : :2::2:22:2:2::2:22:2: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2: 

_output_shapes
::$ 

_output_shapes

:2:$ 

_output_shapes

:22: 

_output_shapes
:2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2: 

_output_shapes
::$ 

_output_shapes

:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::$ 

_output_shapes

:2:$ 

_output_shapes

:22: 

_output_shapes
:2:

_output_shapes
: 
к
Њ
while_cond_857292
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_857292___redundant_placeholder04
0while_while_cond_857292___redundant_placeholder14
0while_while_cond_857292___redundant_placeholder24
0while_while_cond_857292___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Љ
э
H__inference_sequential_4_layer_call_and_return_conditional_losses_856575
simple_rnn_1_input%
simple_rnn_1_856562:2!
simple_rnn_1_856564:2%
simple_rnn_1_856566:22 
dense_8_856569:2
dense_8_856571:
identityЂdense_8/StatefulPartitionedCallЂ$simple_rnn_1/StatefulPartitionedCallЃ
$simple_rnn_1/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_1_inputsimple_rnn_1_856562simple_rnn_1_856564simple_rnn_1_856566*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_856318
dense_8/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_1/StatefulPartitionedCall:output:0dense_8_856569dense_8_856571*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_856336w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^dense_8/StatefulPartitionedCall%^simple_rnn_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2L
$simple_rnn_1/StatefulPartitionedCall$simple_rnn_1/StatefulPartitionedCall:_ [
+
_output_shapes
:џџџџџџџџџ
,
_user_specified_namesimple_rnn_1_input

ъ
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_857441

inputs
states_00
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:22
 matmul_1_readvariableop_resource:22
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2G
ReluReluadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ2: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0
л-
Ш
while_body_856963
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0:2G
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:2L
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:22
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_1_matmul_readvariableop_resource:2E
7while_simple_rnn_cell_1_biasadd_readvariableop_resource:2J
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:22Ђ.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_1/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0І
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0У
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Є
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype0О
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Њ
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype0Њ
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ќ
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2w
while/simple_rnn_cell_1/ReluReluwhile/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ћ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_1/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity*while/simple_rnn_cell_1/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2п

while/NoOpNoOp/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ2: : : : : 2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Ј>
Л
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_856318

inputsB
0simple_rnn_cell_1_matmul_readvariableop_resource:2?
1simple_rnn_cell_1_biasadd_readvariableop_resource:2D
2simple_rnn_cell_1_matmul_1_readvariableop_resource:22
identityЂ(simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_1/MatMul/ReadVariableOpЂ)simple_rnn_cell_1/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0Ќ
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype0
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2k
simple_rnn_cell_1/ReluRelusimple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_856251*
condR
while_cond_856250*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ2*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2Я
NoOpNoOp)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Й
-__inference_simple_rnn_1_layer_call_fn_856887
inputs_0
unknown:2
	unknown_0:2
	unknown_1:22
identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_856032o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
"
б
while_body_855968
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
 while_simple_rnn_cell_1_855990_0:2.
 while_simple_rnn_cell_1_855992_0:22
 while_simple_rnn_cell_1_855994_0:22
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_1_855990:2,
while_simple_rnn_cell_1_855992:20
while_simple_rnn_cell_1_855994:22Ђ/while/simple_rnn_cell_1/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ђ
/while/simple_rnn_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2 while_simple_rnn_cell_1_855990_0 while_simple_rnn_cell_1_855992_0 while_simple_rnn_cell_1_855994_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_855954r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:08while/simple_rnn_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity8while/simple_rnn_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2~

while/NoOpNoOp0^while/simple_rnn_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"B
while_simple_rnn_cell_1_855990 while_simple_rnn_cell_1_855990_0"B
while_simple_rnn_cell_1_855992 while_simple_rnn_cell_1_855992_0"B
while_simple_rnn_cell_1_855994 while_simple_rnn_cell_1_855994_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ2: : : : : 2b
/while/simple_rnn_cell_1/StatefulPartitionedCall/while/simple_rnn_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
в
э
-__inference_sequential_4_layer_call_fn_856629

inputs
unknown:2
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_856343o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Й

к
2__inference_simple_rnn_cell_1_layer_call_fn_857407

inputs
states_0
unknown:2
	unknown_0:2
	unknown_1:22
identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_856076o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0
і
љ
-__inference_sequential_4_layer_call_fn_856559
simple_rnn_1_input
unknown:2
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_856531o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:џџџџџџџџџ
,
_user_specified_namesimple_rnn_1_input
Ц	
є
C__inference_dense_8_layer_call_and_return_conditional_losses_857379

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
к
Њ
while_cond_855967
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_855967___redundant_placeholder04
0while_while_cond_855967___redundant_placeholder14
0while_while_cond_855967___redundant_placeholder24
0while_while_cond_855967___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Ц	
є
C__inference_dense_8_layer_call_and_return_conditional_losses_856336

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
R
Ш
H__inference_sequential_4_layer_call_and_return_conditional_losses_856760

inputsO
=simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resource:2L
>simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resource:2Q
?simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource:228
&dense_8_matmul_readvariableop_resource:25
'dense_8_biasadd_readvariableop_resource:
identityЂdense_8/BiasAdd/ReadVariableOpЂdense_8/MatMul/ReadVariableOpЂ5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOpЂ6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOpЂsimple_rnn_1/whileH
simple_rnn_1/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_1/strided_sliceStridedSlicesimple_rnn_1/Shape:output:0)simple_rnn_1/strided_slice/stack:output:0+simple_rnn_1/strided_slice/stack_1:output:0+simple_rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_1/zeros/packedPack#simple_rnn_1/strided_slice:output:0$simple_rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
simple_rnn_1/zerosFill"simple_rnn_1/zeros/packed:output:0!simple_rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2p
simple_rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
simple_rnn_1/transpose	Transposeinputs$simple_rnn_1/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ^
simple_rnn_1/Shape_1Shapesimple_rnn_1/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_1/strided_slice_1StridedSlicesimple_rnn_1/Shape_1:output:0+simple_rnn_1/strided_slice_1/stack:output:0-simple_rnn_1/strided_slice_1/stack_1:output:0-simple_rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџл
simple_rnn_1/TensorArrayV2TensorListReserve1simple_rnn_1/TensorArrayV2/element_shape:output:0%simple_rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Bsimple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
4simple_rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_1/transpose:y:0Ksimple_rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвl
"simple_rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
simple_rnn_1/strided_slice_2StridedSlicesimple_rnn_1/transpose:y:0+simple_rnn_1/strided_slice_2/stack:output:0-simple_rnn_1/strided_slice_2/stack_1:output:0-simple_rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskВ
4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp=simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0Ц
%simple_rnn_1/simple_rnn_cell_1/MatMulMatMul%simple_rnn_1/strided_slice_2:output:0<simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2А
5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0г
&simple_rnn_1/simple_rnn_cell_1/BiasAddBiasAdd/simple_rnn_1/simple_rnn_cell_1/MatMul:product:0=simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ж
6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype0Р
'simple_rnn_1/simple_rnn_cell_1/MatMul_1MatMulsimple_rnn_1/zeros:output:0>simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2С
"simple_rnn_1/simple_rnn_cell_1/addAddV2/simple_rnn_1/simple_rnn_cell_1/BiasAdd:output:01simple_rnn_1/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
#simple_rnn_1/simple_rnn_cell_1/ReluRelu&simple_rnn_1/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2{
*simple_rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   k
)simple_rnn_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ь
simple_rnn_1/TensorArrayV2_1TensorListReserve3simple_rnn_1/TensorArrayV2_1/element_shape:output:02simple_rnn_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвS
simple_rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџa
simple_rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
simple_rnn_1/whileWhile(simple_rnn_1/while/loop_counter:output:0.simple_rnn_1/while/maximum_iterations:output:0simple_rnn_1/time:output:0%simple_rnn_1/TensorArrayV2_1:handle:0simple_rnn_1/zeros:output:0%simple_rnn_1/strided_slice_1:output:0Dsimple_rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_1_simple_rnn_cell_1_matmul_readvariableop_resource>simple_rnn_1_simple_rnn_cell_1_biasadd_readvariableop_resource?simple_rnn_1_simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( **
body"R 
simple_rnn_1_while_body_856687**
cond"R 
simple_rnn_1_while_cond_856686*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 
=simple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   §
/simple_rnn_1/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_1/while:output:3Fsimple_rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ2*
element_dtype0*
num_elementsu
"simple_rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџn
$simple_rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ш
simple_rnn_1/strided_slice_3StridedSlice8simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_1/strided_slice_3/stack:output:0-simple_rnn_1/strided_slice_3/stack_1:output:0-simple_rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maskr
simple_rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
simple_rnn_1/transpose_1	Transpose8simple_rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
dense_8/MatMulMatMul%simple_rnn_1/strided_slice_3:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџg
IdentityIdentitydense_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџФ
NoOpNoOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp6^simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp5^simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp7^simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp^simple_rnn_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2n
5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp5simple_rnn_1/simple_rnn_cell_1/BiasAdd/ReadVariableOp2l
4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp4simple_rnn_1/simple_rnn_cell_1/MatMul/ReadVariableOp2p
6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp6simple_rnn_1/simple_rnn_cell_1/MatMul_1/ReadVariableOp2(
simple_rnn_1/whilesimple_rnn_1/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ќ9
Ю
simple_rnn_1_while_body_8566876
2simple_rnn_1_while_simple_rnn_1_while_loop_counter<
8simple_rnn_1_while_simple_rnn_1_while_maximum_iterations"
simple_rnn_1_while_placeholder$
 simple_rnn_1_while_placeholder_1$
 simple_rnn_1_while_placeholder_25
1simple_rnn_1_while_simple_rnn_1_strided_slice_1_0q
msimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0:2T
Fsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:2Y
Gsimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:22
simple_rnn_1_while_identity!
simple_rnn_1_while_identity_1!
simple_rnn_1_while_identity_2!
simple_rnn_1_while_identity_3!
simple_rnn_1_while_identity_43
/simple_rnn_1_while_simple_rnn_1_strided_slice_1o
ksimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource:2R
Dsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource:2W
Esimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource:22Ђ;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpЂ<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp
Dsimple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ч
6simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_1_while_placeholderMsimple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Р
:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0ъ
+simple_rnn_1/while/simple_rnn_cell_1/MatMulMatMul=simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2О
;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype0х
,simple_rnn_1/while/simple_rnn_cell_1/BiasAddBiasAdd5simple_rnn_1/while/simple_rnn_cell_1/MatMul:product:0Csimple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ф
<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype0б
-simple_rnn_1/while/simple_rnn_cell_1/MatMul_1MatMul simple_rnn_1_while_placeholder_2Dsimple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2г
(simple_rnn_1/while/simple_rnn_cell_1/addAddV25simple_rnn_1/while/simple_rnn_cell_1/BiasAdd:output:07simple_rnn_1/while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
)simple_rnn_1/while/simple_rnn_cell_1/ReluRelu,simple_rnn_1/while/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
=simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Џ
7simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_1_while_placeholder_1Fsimple_rnn_1/while/TensorArrayV2Write/TensorListSetItem/index:output:07simple_rnn_1/while/simple_rnn_cell_1/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвZ
simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_1/while/addAddV2simple_rnn_1_while_placeholder!simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_1/while/add_1AddV22simple_rnn_1_while_simple_rnn_1_while_loop_counter#simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
simple_rnn_1/while/IdentityIdentitysimple_rnn_1/while/add_1:z:0^simple_rnn_1/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_1/while/Identity_1Identity8simple_rnn_1_while_simple_rnn_1_while_maximum_iterations^simple_rnn_1/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_1/while/Identity_2Identitysimple_rnn_1/while/add:z:0^simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ­
simple_rnn_1/while/Identity_3IdentityGsimple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_1/while/NoOp*
T0*
_output_shapes
: Ў
simple_rnn_1/while/Identity_4Identity7simple_rnn_1/while/simple_rnn_cell_1/Relu:activations:0^simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2
simple_rnn_1/while/NoOpNoOp<^simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp=^simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_1_while_identity$simple_rnn_1/while/Identity:output:0"G
simple_rnn_1_while_identity_1&simple_rnn_1/while/Identity_1:output:0"G
simple_rnn_1_while_identity_2&simple_rnn_1/while/Identity_2:output:0"G
simple_rnn_1_while_identity_3&simple_rnn_1/while/Identity_3:output:0"G
simple_rnn_1_while_identity_4&simple_rnn_1/while/Identity_4:output:0"d
/simple_rnn_1_while_simple_rnn_1_strided_slice_11simple_rnn_1_while_simple_rnn_1_strided_slice_1_0"
Dsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resourceFsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"
Esimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resourceGsimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"
Csimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resourceEsimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0"м
ksimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensormsimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ2: : : : : 2z
;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2x
:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp2|
<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
к
Њ
while_cond_856962
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_856962___redundant_placeholder04
0while_while_cond_856962___redundant_placeholder14
0while_while_cond_856962___redundant_placeholder24
0while_while_cond_856962___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
К

Ё
simple_rnn_1_while_cond_8566866
2simple_rnn_1_while_simple_rnn_1_while_loop_counter<
8simple_rnn_1_while_simple_rnn_1_while_maximum_iterations"
simple_rnn_1_while_placeholder$
 simple_rnn_1_while_placeholder_1$
 simple_rnn_1_while_placeholder_28
4simple_rnn_1_while_less_simple_rnn_1_strided_slice_1N
Jsimple_rnn_1_while_simple_rnn_1_while_cond_856686___redundant_placeholder0N
Jsimple_rnn_1_while_simple_rnn_1_while_cond_856686___redundant_placeholder1N
Jsimple_rnn_1_while_simple_rnn_1_while_cond_856686___redundant_placeholder2N
Jsimple_rnn_1_while_simple_rnn_1_while_cond_856686___redundant_placeholder3
simple_rnn_1_while_identity

simple_rnn_1/while/LessLesssimple_rnn_1_while_placeholder4simple_rnn_1_while_less_simple_rnn_1_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_1/while/IdentityIdentitysimple_rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_1_while_identity$simple_rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
к
Њ
while_cond_856128
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_856128___redundant_placeholder04
0while_while_cond_856128___redundant_placeholder14
0while_while_cond_856128___redundant_placeholder24
0while_while_cond_856128___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Ј>
Л
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_856489

inputsB
0simple_rnn_cell_1_matmul_readvariableop_resource:2?
1simple_rnn_cell_1_biasadd_readvariableop_resource:2D
2simple_rnn_cell_1_matmul_1_readvariableop_resource:22
identityЂ(simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_1/MatMul/ReadVariableOpЂ)simple_rnn_cell_1/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0Ќ
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype0
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2k
simple_rnn_cell_1/ReluRelusimple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_856422*
condR
while_cond_856421*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ2*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2Я
NoOpNoOp)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ц
№
$__inference_signature_wrapper_856614
simple_rnn_1_input
unknown:2
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_855906o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:џџџџџџџџџ
,
_user_specified_namesimple_rnn_1_input
к
Њ
while_cond_857182
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_857182___redundant_placeholder04
0while_while_cond_857182___redundant_placeholder14
0while_while_cond_857182___redundant_placeholder24
0while_while_cond_857182___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
ќ9
Ю
simple_rnn_1_while_body_8568036
2simple_rnn_1_while_simple_rnn_1_while_loop_counter<
8simple_rnn_1_while_simple_rnn_1_while_maximum_iterations"
simple_rnn_1_while_placeholder$
 simple_rnn_1_while_placeholder_1$
 simple_rnn_1_while_placeholder_25
1simple_rnn_1_while_simple_rnn_1_strided_slice_1_0q
msimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0:2T
Fsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:2Y
Gsimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:22
simple_rnn_1_while_identity!
simple_rnn_1_while_identity_1!
simple_rnn_1_while_identity_2!
simple_rnn_1_while_identity_3!
simple_rnn_1_while_identity_43
/simple_rnn_1_while_simple_rnn_1_strided_slice_1o
ksimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource:2R
Dsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource:2W
Esimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource:22Ђ;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpЂ<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp
Dsimple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ч
6simple_rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_1_while_placeholderMsimple_rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Р
:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0ъ
+simple_rnn_1/while/simple_rnn_cell_1/MatMulMatMul=simple_rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2О
;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype0х
,simple_rnn_1/while/simple_rnn_cell_1/BiasAddBiasAdd5simple_rnn_1/while/simple_rnn_cell_1/MatMul:product:0Csimple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ф
<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype0б
-simple_rnn_1/while/simple_rnn_cell_1/MatMul_1MatMul simple_rnn_1_while_placeholder_2Dsimple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2г
(simple_rnn_1/while/simple_rnn_cell_1/addAddV25simple_rnn_1/while/simple_rnn_cell_1/BiasAdd:output:07simple_rnn_1/while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
)simple_rnn_1/while/simple_rnn_cell_1/ReluRelu,simple_rnn_1/while/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
=simple_rnn_1/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Џ
7simple_rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_1_while_placeholder_1Fsimple_rnn_1/while/TensorArrayV2Write/TensorListSetItem/index:output:07simple_rnn_1/while/simple_rnn_cell_1/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвZ
simple_rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_1/while/addAddV2simple_rnn_1_while_placeholder!simple_rnn_1/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_1/while/add_1AddV22simple_rnn_1_while_simple_rnn_1_while_loop_counter#simple_rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
simple_rnn_1/while/IdentityIdentitysimple_rnn_1/while/add_1:z:0^simple_rnn_1/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_1/while/Identity_1Identity8simple_rnn_1_while_simple_rnn_1_while_maximum_iterations^simple_rnn_1/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_1/while/Identity_2Identitysimple_rnn_1/while/add:z:0^simple_rnn_1/while/NoOp*
T0*
_output_shapes
: ­
simple_rnn_1/while/Identity_3IdentityGsimple_rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_1/while/NoOp*
T0*
_output_shapes
: Ў
simple_rnn_1/while/Identity_4Identity7simple_rnn_1/while/simple_rnn_cell_1/Relu:activations:0^simple_rnn_1/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2
simple_rnn_1/while/NoOpNoOp<^simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp=^simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_1_while_identity$simple_rnn_1/while/Identity:output:0"G
simple_rnn_1_while_identity_1&simple_rnn_1/while/Identity_1:output:0"G
simple_rnn_1_while_identity_2&simple_rnn_1/while/Identity_2:output:0"G
simple_rnn_1_while_identity_3&simple_rnn_1/while/Identity_3:output:0"G
simple_rnn_1_while_identity_4&simple_rnn_1/while/Identity_4:output:0"d
/simple_rnn_1_while_simple_rnn_1_strided_slice_11simple_rnn_1_while_simple_rnn_1_strided_slice_1_0"
Dsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resourceFsimple_rnn_1_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"
Esimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resourceGsimple_rnn_1_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"
Csimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resourceEsimple_rnn_1_while_simple_rnn_cell_1_matmul_readvariableop_resource_0"м
ksimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensormsimple_rnn_1_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ2: : : : : 2z
;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp;simple_rnn_1/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2x
:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp:simple_rnn_1/while/simple_rnn_cell_1/MatMul/ReadVariableOp2|
<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp<simple_rnn_1/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
л-
Ш
while_body_856251
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0:2G
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:2L
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:22
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_1_matmul_readvariableop_resource:2E
7while_simple_rnn_cell_1_biasadd_readvariableop_resource:2J
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:22Ђ.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_1/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0І
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0У
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Є
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype0О
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Њ
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype0Њ
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ќ
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2w
while/simple_rnn_cell_1/ReluReluwhile/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ћ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_1/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity*while/simple_rnn_cell_1/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2п

while/NoOpNoOp/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ2: : : : : 2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: "Е	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ф
serving_defaultА
U
simple_rnn_1_input?
$serving_default_simple_rnn_1_input:0џџџџџџџџџ;
dense_80
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Ќ
Д
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures"
_tf_keras_sequential
У
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec"
_tf_keras_rnn_layer
Л
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
non_trainable_variables

 layers
!metrics
"layer_regularization_losses
#layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
щ
$trace_0
%trace_1
&trace_2
'trace_32ў
-__inference_sequential_4_layer_call_fn_856356
-__inference_sequential_4_layer_call_fn_856629
-__inference_sequential_4_layer_call_fn_856644
-__inference_sequential_4_layer_call_fn_856559П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z$trace_0z%trace_1z&trace_2z'trace_3
е
(trace_0
)trace_1
*trace_2
+trace_32ъ
H__inference_sequential_4_layer_call_and_return_conditional_losses_856760
H__inference_sequential_4_layer_call_and_return_conditional_losses_856876
H__inference_sequential_4_layer_call_and_return_conditional_losses_856575
H__inference_sequential_4_layer_call_and_return_conditional_losses_856591П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z(trace_0z)trace_1z*trace_2z+trace_3
зBд
!__inference__wrapped_model_855906simple_rnn_1_input"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
­
,iter

-beta_1

.beta_2
	/decay
0learning_ratemambmcmdmevfvgvhvivj"
	optimizer
,
1serving_default"
signature_map
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
Й

2states
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ў
8trace_0
9trace_1
:trace_2
;trace_32
-__inference_simple_rnn_1_layer_call_fn_856887
-__inference_simple_rnn_1_layer_call_fn_856898
-__inference_simple_rnn_1_layer_call_fn_856909
-__inference_simple_rnn_1_layer_call_fn_856920д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z8trace_0z9trace_1z:trace_2z;trace_3
ъ
<trace_0
=trace_1
>trace_2
?trace_32џ
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_857030
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_857140
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_857250
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_857360д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z<trace_0z=trace_1z>trace_2z?trace_3
ш
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
F_random_generator

kernel
recurrent_kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ь
Ltrace_02Я
(__inference_dense_8_layer_call_fn_857369Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zLtrace_0

Mtrace_02ъ
C__inference_dense_8_layer_call_and_return_conditional_losses_857379Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zMtrace_0
 :22dense_8/kernel
:2dense_8/bias
7:522%simple_rnn_1/simple_rnn_cell_1/kernel
A:?222/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel
1:/22#simple_rnn_1/simple_rnn_cell_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
-__inference_sequential_4_layer_call_fn_856356simple_rnn_1_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
-__inference_sequential_4_layer_call_fn_856629inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
-__inference_sequential_4_layer_call_fn_856644inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
-__inference_sequential_4_layer_call_fn_856559simple_rnn_1_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
H__inference_sequential_4_layer_call_and_return_conditional_losses_856760inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
H__inference_sequential_4_layer_call_and_return_conditional_losses_856876inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЅBЂ
H__inference_sequential_4_layer_call_and_return_conditional_losses_856575simple_rnn_1_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЅBЂ
H__inference_sequential_4_layer_call_and_return_conditional_losses_856591simple_rnn_1_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
жBг
$__inference_signature_wrapper_856614simple_rnn_1_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
-__inference_simple_rnn_1_layer_call_fn_856887inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
-__inference_simple_rnn_1_layer_call_fn_856898inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
-__inference_simple_rnn_1_layer_call_fn_856909inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
-__inference_simple_rnn_1_layer_call_fn_856920inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
АB­
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_857030inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
АB­
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_857140inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЎBЋ
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_857250inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЎBЋ
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_857360inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
п
Utrace_0
Vtrace_12Ј
2__inference_simple_rnn_cell_1_layer_call_fn_857393
2__inference_simple_rnn_cell_1_layer_call_fn_857407Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zUtrace_0zVtrace_1

Wtrace_0
Xtrace_12о
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_857424
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_857441Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zWtrace_0zXtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
мBй
(__inference_dense_8_layer_call_fn_857369inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_dense_8_layer_call_and_return_conditional_losses_857379inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
N
Y	variables
Z	keras_api
	[total
	\count"
_tf_keras_metric
N
]	variables
^	keras_api
	_total
	`count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
2__inference_simple_rnn_cell_1_layer_call_fn_857393inputsstates/0"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
2__inference_simple_rnn_cell_1_layer_call_fn_857407inputsstates/0"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ІBЃ
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_857424inputsstates/0"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ІBЃ
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_857441inputsstates/0"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
[0
\1"
trackable_list_wrapper
-
Y	variables"
_generic_user_object
:  (2total
:  (2count
.
_0
`1"
trackable_list_wrapper
-
]	variables"
_generic_user_object
:  (2total
:  (2count
%:#22Adam/dense_8/kernel/m
:2Adam/dense_8/bias/m
<::22,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/m
F:D2226Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/m
6:422*Adam/simple_rnn_1/simple_rnn_cell_1/bias/m
%:#22Adam/dense_8/kernel/v
:2Adam/dense_8/bias/v
<::22,Adam/simple_rnn_1/simple_rnn_cell_1/kernel/v
F:D2226Adam/simple_rnn_1/simple_rnn_cell_1/recurrent_kernel/v
6:422*Adam/simple_rnn_1/simple_rnn_cell_1/bias/v 
!__inference__wrapped_model_855906{?Ђ<
5Ђ2
0-
simple_rnn_1_inputџџџџџџџџџ
Њ "1Њ.
,
dense_8!
dense_8џџџџџџџџџЃ
C__inference_dense_8_layer_call_and_return_conditional_losses_857379\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "%Ђ"

0џџџџџџџџџ
 {
(__inference_dense_8_layer_call_fn_857369O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "џџџџџџџџџУ
H__inference_sequential_4_layer_call_and_return_conditional_losses_856575wGЂD
=Ђ:
0-
simple_rnn_1_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 У
H__inference_sequential_4_layer_call_and_return_conditional_losses_856591wGЂD
=Ђ:
0-
simple_rnn_1_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 З
H__inference_sequential_4_layer_call_and_return_conditional_losses_856760k;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 З
H__inference_sequential_4_layer_call_and_return_conditional_losses_856876k;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
-__inference_sequential_4_layer_call_fn_856356jGЂD
=Ђ:
0-
simple_rnn_1_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
-__inference_sequential_4_layer_call_fn_856559jGЂD
=Ђ:
0-
simple_rnn_1_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
-__inference_sequential_4_layer_call_fn_856629^;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
-__inference_sequential_4_layer_call_fn_856644^;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџК
$__inference_signature_wrapper_856614UЂR
Ђ 
KЊH
F
simple_rnn_1_input0-
simple_rnn_1_inputџџџџџџџџџ"1Њ.
,
dense_8!
dense_8џџџџџџџџџЩ
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_857030}OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 Щ
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_857140}OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 Й
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_857250m?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 Й
H__inference_simple_rnn_1_layer_call_and_return_conditional_losses_857360m?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 Ё
-__inference_simple_rnn_1_layer_call_fn_856887pOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2Ё
-__inference_simple_rnn_1_layer_call_fn_856898pOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2
-__inference_simple_rnn_1_layer_call_fn_856909`?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2
-__inference_simple_rnn_1_layer_call_fn_856920`?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_857424З\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ2
p 
Њ "RЂO
HЂE

0/0џџџџџџџџџ2
$!

0/1/0џџџџџџџџџ2
 
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_857441З\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ2
p
Њ "RЂO
HЂE

0/0џџџџџџџџџ2
$!

0/1/0џџџџџџџџџ2
 р
2__inference_simple_rnn_cell_1_layer_call_fn_857393Љ\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ2
p 
Њ "DЂA

0џџџџџџџџџ2
"

1/0џџџџџџџџџ2р
2__inference_simple_rnn_cell_1_layer_call_fn_857407Љ\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ2
p
Њ "DЂA

0џџџџџџџџџ2
"

1/0џџџџџџџџџ2