Еє
Чч
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
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
dtypetypeИ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
┴
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
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
-
Tanh
x"T
y"T"
Ttype:

2
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleКщшelement_dtype"
element_dtypetype"

shape_typetype:
2	
Я
TensorListReserve
element_shape"
shape_type
num_elements(
handleКщшelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
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
И"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8зз
м
*Adam/simple_rnn_2/simple_rnn_cell_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/simple_rnn_2/simple_rnn_cell_4/bias/v
е
>Adam/simple_rnn_2/simple_rnn_cell_4/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_2/simple_rnn_cell_4/bias/v*
_output_shapes
:@*
dtype0
╚
6Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*G
shared_name86Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/v
┴
JAdam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
┤
,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*=
shared_name.,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/v
н
@Adam/simple_rnn_2/simple_rnn_cell_4/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/v*
_output_shapes

:@*
dtype0
А
Adam/dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_17/bias/v
y
(Adam/dense_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/v*
_output_shapes
:*
dtype0
И
Adam/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_17/kernel/v
Б
*Adam/dense_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/v*
_output_shapes

:@*
dtype0
м
*Adam/simple_rnn_2/simple_rnn_cell_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/simple_rnn_2/simple_rnn_cell_4/bias/m
е
>Adam/simple_rnn_2/simple_rnn_cell_4/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_2/simple_rnn_cell_4/bias/m*
_output_shapes
:@*
dtype0
╚
6Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*G
shared_name86Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/m
┴
JAdam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
┤
,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*=
shared_name.,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/m
н
@Adam/simple_rnn_2/simple_rnn_cell_4/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/m*
_output_shapes

:@*
dtype0
А
Adam/dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_17/bias/m
y
(Adam/dense_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/m*
_output_shapes
:*
dtype0
И
Adam/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_17/kernel/m
Б
*Adam/dense_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/m*
_output_shapes

:@*
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
Ю
#simple_rnn_2/simple_rnn_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#simple_rnn_2/simple_rnn_cell_4/bias
Ч
7simple_rnn_2/simple_rnn_cell_4/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_2/simple_rnn_cell_4/bias*
_output_shapes
:@*
dtype0
║
/simple_rnn_2/simple_rnn_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*@
shared_name1/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel
│
Csimple_rnn_2/simple_rnn_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel*
_output_shapes

:@@*
dtype0
ж
%simple_rnn_2/simple_rnn_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*6
shared_name'%simple_rnn_2/simple_rnn_cell_4/kernel
Я
9simple_rnn_2/simple_rnn_cell_4/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_2/simple_rnn_cell_4/kernel*
_output_shapes

:@*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:*
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:@*
dtype0
В
serving_default_input_8Placeholder*+
_output_shapes
:         *
dtype0* 
shape:         
┌
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_8%simple_rnn_2/simple_rnn_cell_4/kernel#simple_rnn_2/simple_rnn_cell_4/bias/simple_rnn_2/simple_rnn_cell_4/recurrent_kerneldense_17/kerneldense_17/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_1292972

NoOpNoOp
╙2
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*О2
valueД2BБ2 B·1
з
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*
к
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec*
е
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator* 
ж
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias*
'
$0
%1
&2
"3
#4*
'
$0
%1
&2
"3
#4*
* 
░
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
,trace_0
-trace_1
.trace_2
/trace_3* 
6
0trace_0
1trace_1
2trace_2
3trace_3* 
* 
Ю
4iter

5beta_1

6beta_2
	7decay
8learning_rate"mr#ms$mt%mu&mv"vw#vx$vy%vz&v{*

9serving_default* 

$0
%1
&2*

$0
%1
&2*
* 
Я

:states
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
@trace_0
Atrace_1
Btrace_2
Ctrace_3* 
6
Dtrace_0
Etrace_1
Ftrace_2
Gtrace_3* 
╙
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
N_random_generator

$kernel
%recurrent_kernel
&bias*
* 
* 
* 
* 
С
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Ttrace_0
Utrace_1* 

Vtrace_0
Wtrace_1* 
* 

"0
#1*

"0
#1*
* 
У
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

]trace_0* 

^trace_0* 
_Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_17/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_2/simple_rnn_cell_4/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#simple_rnn_2/simple_rnn_cell_4/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

_0
`1*
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

0*
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
$0
%1
&2*

$0
%1
&2*
* 
У
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

ftrace_0
gtrace_1* 

htrace_0
itrace_1* 
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
* 
* 
* 
* 
* 
* 
8
j	variables
k	keras_api
	ltotal
	mcount*
8
n	variables
o	keras_api
	ptotal
	qcount*
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
l0
m1*

j	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

p0
q1*

n	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_17/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_17/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
УМ
VARIABLE_VALUE6Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUE*Adam/simple_rnn_2/simple_rnn_cell_4/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_17/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_17/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
УМ
VARIABLE_VALUE6Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUE*Adam/simple_rnn_2/simple_rnn_cell_4/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ц
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOp9simple_rnn_2/simple_rnn_cell_4/kernel/Read/ReadVariableOpCsimple_rnn_2/simple_rnn_cell_4/recurrent_kernel/Read/ReadVariableOp7simple_rnn_2/simple_rnn_cell_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_17/kernel/m/Read/ReadVariableOp(Adam/dense_17/bias/m/Read/ReadVariableOp@Adam/simple_rnn_2/simple_rnn_cell_4/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_2/simple_rnn_cell_4/bias/m/Read/ReadVariableOp*Adam/dense_17/kernel/v/Read/ReadVariableOp(Adam/dense_17/bias/v/Read/ReadVariableOp@Adam/simple_rnn_2/simple_rnn_cell_4/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_2/simple_rnn_cell_4/bias/v/Read/ReadVariableOpConst*%
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
GPU 2J 8В *)
f$R"
 __inference__traced_save_1293930
▒
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_17/kerneldense_17/bias%simple_rnn_2/simple_rnn_cell_4/kernel/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel#simple_rnn_2/simple_rnn_cell_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_17/kernel/mAdam/dense_17/bias/m,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/m6Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/m*Adam/simple_rnn_2/simple_rnn_cell_4/bias/mAdam/dense_17/kernel/vAdam/dense_17/bias/v,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/v6Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/v*Adam/simple_rnn_2/simple_rnn_cell_4/bias/v*$
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
GPU 2J 8В *,
f'R%
#__inference__traced_restore_1294012ай
л>
╝
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1293727

inputsB
0simple_rnn_cell_4_matmul_readvariableop_resource:@?
1simple_rnn_cell_4_biasadd_readvariableop_resource:@D
2simple_rnn_cell_4_matmul_1_readvariableop_resource:@@
identityИв(simple_rnn_cell_4/BiasAdd/ReadVariableOpв'simple_rnn_cell_4/MatMul/ReadVariableOpв)simple_rnn_cell_4/MatMul_1/ReadVariableOpвwhile;
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
valueB:╤
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
value	B :@s
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
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
:         *
shrink_axis_maskШ
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Я
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ц
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0м
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ь
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Щ
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ъ
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @k
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┌
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1293660*
condR
while_cond_1293659*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @╧
NoOpNoOp)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╬>
╛
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1293507
inputs_0B
0simple_rnn_cell_4_matmul_readvariableop_resource:@?
1simple_rnn_cell_4_biasadd_readvariableop_resource:@D
2simple_rnn_cell_4_matmul_1_readvariableop_resource:@@
identityИв(simple_rnn_cell_4/BiasAdd/ReadVariableOpв'simple_rnn_cell_4/MatMul/ReadVariableOpв)simple_rnn_cell_4/MatMul_1/ReadVariableOpвwhile=
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
valueB:╤
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
value	B :@s
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
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
:         *
shrink_axis_maskШ
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Я
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ц
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0м
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ь
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Щ
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ъ
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @k
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┌
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1293440*
condR
while_cond_1293439*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @╧
NoOpNoOp)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
Д
ы
N__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1293835

inputs
states_00
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:         @G
TanhTanhadd:z:0*
T0*'
_output_shapes
:         @W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:         @Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:         @С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :         @: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         @
"
_user_specified_name
states/0
┘
d
F__inference_dropout_4_layer_call_and_return_conditional_losses_1293742

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         @[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Д
ы
N__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1293818

inputs
states_00
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:         @G
TanhTanhadd:z:0*
T0*'
_output_shapes
:         @W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:         @Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:         @С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :         @: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         @
"
_user_specified_name
states/0
╟-
╔
while_body_1293550
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0:@G
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_4_matmul_readvariableop_resource:@E
7while_simple_rnn_cell_4_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:@@Ив.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpв-while/simple_rnn_cell_4/MatMul/ReadVariableOpв/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ж
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0├
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @д
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0╛
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @к
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0к
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @м
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @w
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0 while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype0:щш╥M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: }
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:         @▀

while/NoOpNoOp/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         @: : : : : 2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:

_output_shapes
: :

_output_shapes
: 
▀
п
while_cond_1293549
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1293549___redundant_placeholder05
1while_while_cond_1293549___redundant_placeholder15
1while_while_cond_1293549___redundant_placeholder25
1while_while_cond_1293549___redundant_placeholder3
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
-: : : : :         @: ::::: 
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
:         @:

_output_shapes
: :

_output_shapes
:
г"
╪
while_body_1292293
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
!while_simple_rnn_cell_4_1292315_0:@/
!while_simple_rnn_cell_4_1292317_0:@3
!while_simple_rnn_cell_4_1292319_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
while_simple_rnn_cell_4_1292315:@-
while_simple_rnn_cell_4_1292317:@1
while_simple_rnn_cell_4_1292319:@@Ив/while/simple_rnn_cell_4/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ж
/while/simple_rnn_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_4_1292315_0!while_simple_rnn_cell_4_1292317_0!while_simple_rnn_cell_4_1292319_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1292279r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Й
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:08while/simple_rnn_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Х
while/Identity_4Identity8while/simple_rnn_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         @~

while/NoOpNoOp0^while/simple_rnn_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_4_1292315!while_simple_rnn_cell_4_1292315_0"D
while_simple_rnn_cell_4_1292317!while_simple_rnn_cell_4_1292317_0"D
while_simple_rnn_cell_4_1292319!while_simple_rnn_cell_4_1292319_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         @: : : : : 2b
/while/simple_rnn_cell_4/StatefulPartitionedCall/while/simple_rnn_cell_4/StatefulPartitionedCall: 
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
:         @:

_output_shapes
: :

_output_shapes
: 
Ъ9
╫
 __inference__traced_save_1293930
file_prefix.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableopD
@savev2_simple_rnn_2_simple_rnn_cell_4_kernel_read_readvariableopN
Jsavev2_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_2_simple_rnn_cell_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_17_kernel_m_read_readvariableop3
/savev2_adam_dense_17_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_2_simple_rnn_cell_4_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_2_simple_rnn_cell_4_bias_m_read_readvariableop5
1savev2_adam_dense_17_kernel_v_read_readvariableop3
/savev2_adam_dense_17_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_2_simple_rnn_cell_4_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_2_simple_rnn_cell_4_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: С
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*║
value░BнB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЯ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B ╥
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop@savev2_simple_rnn_2_simple_rnn_cell_4_kernel_read_readvariableopJsavev2_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_read_readvariableop>savev2_simple_rnn_2_simple_rnn_cell_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_17_kernel_m_read_readvariableop/savev2_adam_dense_17_bias_m_read_readvariableopGsavev2_adam_simple_rnn_2_simple_rnn_cell_4_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_2_simple_rnn_cell_4_bias_m_read_readvariableop1savev2_adam_dense_17_kernel_v_read_readvariableop/savev2_adam_dense_17_bias_v_read_readvariableopGsavev2_adam_simple_rnn_2_simple_rnn_cell_4_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_2_simple_rnn_cell_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
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

identity_1Identity_1:output:0*й
_input_shapesЧ
Ф: :@::@:@@:@: : : : : : : : : :@::@:@@:@:@::@:@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:
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

:@: 

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:

_output_shapes
: 
б
G
+__inference_dropout_4_layer_call_fn_1293732

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1292656`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
▀
п
while_cond_1293439
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1293439___redundant_placeholder05
1while_while_cond_1293439___redundant_placeholder15
1while_while_cond_1293439___redundant_placeholder25
1while_while_cond_1293439___redundant_placeholder3
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
-: : : : :         @: ::::: 
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
:         @:

_output_shapes
: :

_output_shapes
:
з
ц
%__inference_signature_wrapper_1292972
input_8
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__wrapped_model_1292231o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:         
!
_user_specified_name	input_8
л>
╝
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1292844

inputsB
0simple_rnn_cell_4_matmul_readvariableop_resource:@?
1simple_rnn_cell_4_biasadd_readvariableop_resource:@D
2simple_rnn_cell_4_matmul_1_readvariableop_resource:@@
identityИв(simple_rnn_cell_4/BiasAdd/ReadVariableOpв'simple_rnn_cell_4/MatMul/ReadVariableOpв)simple_rnn_cell_4/MatMul_1/ReadVariableOpвwhile;
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
valueB:╤
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
value	B :@s
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
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
:         *
shrink_axis_maskШ
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Я
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ц
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0м
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ь
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Щ
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ъ
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @k
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┌
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1292777*
condR
while_cond_1292776*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @╧
NoOpNoOp)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╟-
╔
while_body_1292576
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0:@G
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_4_matmul_readvariableop_resource:@E
7while_simple_rnn_cell_4_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:@@Ив.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpв-while/simple_rnn_cell_4/MatMul/ReadVariableOpв/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ж
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0├
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @д
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0╛
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @к
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0к
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @м
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @w
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0 while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype0:щш╥M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: }
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:         @▀

while/NoOpNoOp/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         @: : : : : 2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:

_output_shapes
: :

_output_shapes
: 
А
╕
.__inference_simple_rnn_2_layer_call_fn_1293276

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1292643o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
▀
п
while_cond_1293659
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1293659___redundant_placeholder05
1while_while_cond_1293659___redundant_placeholder15
1while_while_cond_1293659___redundant_placeholder25
1while_while_cond_1293659___redundant_placeholder3
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
-: : : : :         @: ::::: 
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
:         @:

_output_shapes
: :

_output_shapes
:
┘
Ё
/__inference_sequential_11_layer_call_fn_1292915
input_8
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_11_layer_call_and_return_conditional_losses_1292887o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:         
!
_user_specified_name	input_8
Е5
Я
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1292357

inputs+
simple_rnn_cell_4_1292280:@'
simple_rnn_cell_4_1292282:@+
simple_rnn_cell_4_1292284:@@
identityИв)simple_rnn_cell_4/StatefulPartitionedCallвwhile;
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
valueB:╤
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
value	B :@s
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
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
:         *
shrink_axis_maskы
)simple_rnn_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_4_1292280simple_rnn_cell_4_1292282simple_rnn_cell_4_1292284*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1292279n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Т
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_4_1292280simple_rnn_cell_4_1292282simple_rnn_cell_4_1292284*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1292293*
condR
while_cond_1292292*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @z
NoOpNoOp*^simple_rnn_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2V
)simple_rnn_cell_4/StatefulPartitionedCall)simple_rnn_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
а
П
J__inference_sequential_11_layer_call_and_return_conditional_losses_1292887

inputs&
simple_rnn_2_1292873:@"
simple_rnn_2_1292875:@&
simple_rnn_2_1292877:@@"
dense_17_1292881:@
dense_17_1292883:
identityИв dense_17/StatefulPartitionedCallв!dropout_4/StatefulPartitionedCallв$simple_rnn_2/StatefulPartitionedCallЫ
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_2_1292873simple_rnn_2_1292875simple_rnn_2_1292877*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1292844Є
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1292718Ч
 dense_17/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_17_1292881dense_17_1292883*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1292668x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ┤
NoOpNoOp!^dense_17/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
л>
╝
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1293617

inputsB
0simple_rnn_cell_4_matmul_readvariableop_resource:@?
1simple_rnn_cell_4_biasadd_readvariableop_resource:@D
2simple_rnn_cell_4_matmul_1_readvariableop_resource:@@
identityИв(simple_rnn_cell_4/BiasAdd/ReadVariableOpв'simple_rnn_cell_4/MatMul/ReadVariableOpв)simple_rnn_cell_4/MatMul_1/ReadVariableOpвwhile;
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
valueB:╤
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
value	B :@s
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
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
:         *
shrink_axis_maskШ
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Я
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ц
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0м
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ь
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Щ
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ъ
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @k
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┌
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1293550*
condR
while_cond_1293549*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @╧
NoOpNoOp)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Ш
║
.__inference_simple_rnn_2_layer_call_fn_1293254
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@@
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1292357o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
Ш
║
.__inference_simple_rnn_2_layer_call_fn_1293265
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@@
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1292518o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
▀
п
while_cond_1292292
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1292292___redundant_placeholder05
1while_while_cond_1292292___redundant_placeholder15
1while_while_cond_1292292___redundant_placeholder25
1while_while_cond_1292292___redundant_placeholder3
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
-: : : : :         @: ::::: 
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
:         @:

_output_shapes
: :

_output_shapes
:
щ9
╧
simple_rnn_2_while_body_12930456
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_25
1simple_rnn_2_while_simple_rnn_2_strided_slice_1_0q
msimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0:@T
Fsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:@Y
Gsimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:@@
simple_rnn_2_while_identity!
simple_rnn_2_while_identity_1!
simple_rnn_2_while_identity_2!
simple_rnn_2_while_identity_3!
simple_rnn_2_while_identity_43
/simple_rnn_2_while_simple_rnn_2_strided_slice_1o
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource:@R
Dsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource:@W
Esimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource:@@Ив;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpв:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOpв<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpХ
Dsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ч
6simple_rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_2_while_placeholderMsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0└
:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0ъ
+simple_rnn_2/while/simple_rnn_cell_4/MatMulMatMul=simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╛
;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0х
,simple_rnn_2/while/simple_rnn_cell_4/BiasAddBiasAdd5simple_rnn_2/while/simple_rnn_cell_4/MatMul:product:0Csimple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @─
<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0╤
-simple_rnn_2/while/simple_rnn_cell_4/MatMul_1MatMul simple_rnn_2_while_placeholder_2Dsimple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╙
(simple_rnn_2/while/simple_rnn_cell_4/addAddV25simple_rnn_2/while/simple_rnn_cell_4/BiasAdd:output:07simple_rnn_2/while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @С
)simple_rnn_2/while/simple_rnn_cell_4/TanhTanh,simple_rnn_2/while/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @
=simple_rnn_2/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : е
7simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_2_while_placeholder_1Fsimple_rnn_2/while/TensorArrayV2Write/TensorListSetItem/index:output:0-simple_rnn_2/while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype0:щш╥Z
simple_rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Г
simple_rnn_2/while/addAddV2simple_rnn_2_while_placeholder!simple_rnn_2/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ы
simple_rnn_2/while/add_1AddV22simple_rnn_2_while_simple_rnn_2_while_loop_counter#simple_rnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: А
simple_rnn_2/while/IdentityIdentitysimple_rnn_2/while/add_1:z:0^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: Ю
simple_rnn_2/while/Identity_1Identity8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: А
simple_rnn_2/while/Identity_2Identitysimple_rnn_2/while/add:z:0^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: н
simple_rnn_2/while/Identity_3IdentityGsimple_rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: д
simple_rnn_2/while/Identity_4Identity-simple_rnn_2/while/simple_rnn_cell_4/Tanh:y:0^simple_rnn_2/while/NoOp*
T0*'
_output_shapes
:         @У
simple_rnn_2/while/NoOpNoOp<^simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0"G
simple_rnn_2_while_identity_1&simple_rnn_2/while/Identity_1:output:0"G
simple_rnn_2_while_identity_2&simple_rnn_2/while/Identity_2:output:0"G
simple_rnn_2_while_identity_3&simple_rnn_2/while/Identity_3:output:0"G
simple_rnn_2_while_identity_4&simple_rnn_2/while/Identity_4:output:0"d
/simple_rnn_2_while_simple_rnn_2_strided_slice_11simple_rnn_2_while_simple_rnn_2_strided_slice_1_0"О
Dsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resourceFsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"Р
Esimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resourceGsimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"М
Csimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resourceEsimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0"▄
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensormsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         @: : : : : 2z
;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2x
:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp2|
<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:

_output_shapes
: :

_output_shapes
: 
√
ь
J__inference_sequential_11_layer_call_and_return_conditional_losses_1292932
input_8&
simple_rnn_2_1292918:@"
simple_rnn_2_1292920:@&
simple_rnn_2_1292922:@@"
dense_17_1292926:@
dense_17_1292928:
identityИв dense_17/StatefulPartitionedCallв$simple_rnn_2/StatefulPartitionedCallЬ
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallinput_8simple_rnn_2_1292918simple_rnn_2_1292920simple_rnn_2_1292922*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1292643т
dropout_4/PartitionedCallPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1292656П
 dense_17/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_17_1292926dense_17_1292928*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1292668x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Р
NoOpNoOp!^dense_17/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall:T P
+
_output_shapes
:         
!
_user_specified_name	input_8
▀
п
while_cond_1292453
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1292453___redundant_placeholder05
1while_while_cond_1292453___redundant_placeholder15
1while_while_cond_1292453___redundant_placeholder25
1while_while_cond_1292453___redundant_placeholder3
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
-: : : : :         @: ::::: 
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
:         @:

_output_shapes
: :

_output_shapes
:
ЕG
ў
-sequential_11_simple_rnn_2_while_body_1292157R
Nsequential_11_simple_rnn_2_while_sequential_11_simple_rnn_2_while_loop_counterX
Tsequential_11_simple_rnn_2_while_sequential_11_simple_rnn_2_while_maximum_iterations0
,sequential_11_simple_rnn_2_while_placeholder2
.sequential_11_simple_rnn_2_while_placeholder_12
.sequential_11_simple_rnn_2_while_placeholder_2Q
Msequential_11_simple_rnn_2_while_sequential_11_simple_rnn_2_strided_slice_1_0О
Йsequential_11_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_11_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0e
Ssequential_11_simple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0:@b
Tsequential_11_simple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:@g
Usequential_11_simple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:@@-
)sequential_11_simple_rnn_2_while_identity/
+sequential_11_simple_rnn_2_while_identity_1/
+sequential_11_simple_rnn_2_while_identity_2/
+sequential_11_simple_rnn_2_while_identity_3/
+sequential_11_simple_rnn_2_while_identity_4O
Ksequential_11_simple_rnn_2_while_sequential_11_simple_rnn_2_strided_slice_1М
Зsequential_11_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_11_simple_rnn_2_tensorarrayunstack_tensorlistfromtensorc
Qsequential_11_simple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource:@`
Rsequential_11_simple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource:@e
Ssequential_11_simple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource:@@ИвIsequential_11/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpвHsequential_11/simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOpвJsequential_11/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpг
Rsequential_11/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       о
Dsequential_11/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЙsequential_11_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_11_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0,sequential_11_simple_rnn_2_while_placeholder[sequential_11/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0▄
Hsequential_11/simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOpSsequential_11_simple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Ф
9sequential_11/simple_rnn_2/while/simple_rnn_cell_4/MatMulMatMulKsequential_11/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0Psequential_11/simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @┌
Isequential_11/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOpTsequential_11_simple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0П
:sequential_11/simple_rnn_2/while/simple_rnn_cell_4/BiasAddBiasAddCsequential_11/simple_rnn_2/while/simple_rnn_cell_4/MatMul:product:0Qsequential_11/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @р
Jsequential_11/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOpUsequential_11_simple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0√
;sequential_11/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1MatMul.sequential_11_simple_rnn_2_while_placeholder_2Rsequential_11/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @¤
6sequential_11/simple_rnn_2/while/simple_rnn_cell_4/addAddV2Csequential_11/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd:output:0Esequential_11/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @н
7sequential_11/simple_rnn_2/while/simple_rnn_cell_4/TanhTanh:sequential_11/simple_rnn_2/while/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @Н
Ksequential_11/simple_rnn_2/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ▌
Esequential_11/simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_11_simple_rnn_2_while_placeholder_1Tsequential_11/simple_rnn_2/while/TensorArrayV2Write/TensorListSetItem/index:output:0;sequential_11/simple_rnn_2/while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype0:щш╥h
&sequential_11/simple_rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :н
$sequential_11/simple_rnn_2/while/addAddV2,sequential_11_simple_rnn_2_while_placeholder/sequential_11/simple_rnn_2/while/add/y:output:0*
T0*
_output_shapes
: j
(sequential_11/simple_rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :╙
&sequential_11/simple_rnn_2/while/add_1AddV2Nsequential_11_simple_rnn_2_while_sequential_11_simple_rnn_2_while_loop_counter1sequential_11/simple_rnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: к
)sequential_11/simple_rnn_2/while/IdentityIdentity*sequential_11/simple_rnn_2/while/add_1:z:0&^sequential_11/simple_rnn_2/while/NoOp*
T0*
_output_shapes
: ╓
+sequential_11/simple_rnn_2/while/Identity_1IdentityTsequential_11_simple_rnn_2_while_sequential_11_simple_rnn_2_while_maximum_iterations&^sequential_11/simple_rnn_2/while/NoOp*
T0*
_output_shapes
: к
+sequential_11/simple_rnn_2/while/Identity_2Identity(sequential_11/simple_rnn_2/while/add:z:0&^sequential_11/simple_rnn_2/while/NoOp*
T0*
_output_shapes
: ╫
+sequential_11/simple_rnn_2/while/Identity_3IdentityUsequential_11/simple_rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^sequential_11/simple_rnn_2/while/NoOp*
T0*
_output_shapes
: ╬
+sequential_11/simple_rnn_2/while/Identity_4Identity;sequential_11/simple_rnn_2/while/simple_rnn_cell_4/Tanh:y:0&^sequential_11/simple_rnn_2/while/NoOp*
T0*'
_output_shapes
:         @╦
%sequential_11/simple_rnn_2/while/NoOpNoOpJ^sequential_11/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpI^sequential_11/simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOpK^sequential_11/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "_
)sequential_11_simple_rnn_2_while_identity2sequential_11/simple_rnn_2/while/Identity:output:0"c
+sequential_11_simple_rnn_2_while_identity_14sequential_11/simple_rnn_2/while/Identity_1:output:0"c
+sequential_11_simple_rnn_2_while_identity_24sequential_11/simple_rnn_2/while/Identity_2:output:0"c
+sequential_11_simple_rnn_2_while_identity_34sequential_11/simple_rnn_2/while/Identity_3:output:0"c
+sequential_11_simple_rnn_2_while_identity_44sequential_11/simple_rnn_2/while/Identity_4:output:0"Ь
Ksequential_11_simple_rnn_2_while_sequential_11_simple_rnn_2_strided_slice_1Msequential_11_simple_rnn_2_while_sequential_11_simple_rnn_2_strided_slice_1_0"к
Rsequential_11_simple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resourceTsequential_11_simple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"м
Ssequential_11_simple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resourceUsequential_11_simple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"и
Qsequential_11_simple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resourceSsequential_11_simple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0"Ц
Зsequential_11_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_11_simple_rnn_2_tensorarrayunstack_tensorlistfromtensorЙsequential_11_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_11_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         @: : : : : 2Ц
Isequential_11/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpIsequential_11/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2Ф
Hsequential_11/simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOpHsequential_11/simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp2Ш
Jsequential_11/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpJsequential_11/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:

_output_shapes
: :

_output_shapes
: 
┘
Ё
/__inference_sequential_11_layer_call_fn_1292688
input_8
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_11_layer_call_and_return_conditional_losses_1292675o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:         
!
_user_specified_name	input_8
╬>
╛
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1293397
inputs_0B
0simple_rnn_cell_4_matmul_readvariableop_resource:@?
1simple_rnn_cell_4_biasadd_readvariableop_resource:@D
2simple_rnn_cell_4_matmul_1_readvariableop_resource:@@
identityИв(simple_rnn_cell_4/BiasAdd/ReadVariableOpв'simple_rnn_cell_4/MatMul/ReadVariableOpв)simple_rnn_cell_4/MatMul_1/ReadVariableOpвwhile=
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
valueB:╤
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
value	B :@s
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
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
:         *
shrink_axis_maskШ
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Я
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ц
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0м
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ь
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Щ
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ъ
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @k
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┌
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1293330*
condR
while_cond_1293329*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @╧
NoOpNoOp)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
║
░
-sequential_11_simple_rnn_2_while_cond_1292156R
Nsequential_11_simple_rnn_2_while_sequential_11_simple_rnn_2_while_loop_counterX
Tsequential_11_simple_rnn_2_while_sequential_11_simple_rnn_2_while_maximum_iterations0
,sequential_11_simple_rnn_2_while_placeholder2
.sequential_11_simple_rnn_2_while_placeholder_12
.sequential_11_simple_rnn_2_while_placeholder_2T
Psequential_11_simple_rnn_2_while_less_sequential_11_simple_rnn_2_strided_slice_1k
gsequential_11_simple_rnn_2_while_sequential_11_simple_rnn_2_while_cond_1292156___redundant_placeholder0k
gsequential_11_simple_rnn_2_while_sequential_11_simple_rnn_2_while_cond_1292156___redundant_placeholder1k
gsequential_11_simple_rnn_2_while_sequential_11_simple_rnn_2_while_cond_1292156___redundant_placeholder2k
gsequential_11_simple_rnn_2_while_sequential_11_simple_rnn_2_while_cond_1292156___redundant_placeholder3-
)sequential_11_simple_rnn_2_while_identity
╬
%sequential_11/simple_rnn_2/while/LessLess,sequential_11_simple_rnn_2_while_placeholderPsequential_11_simple_rnn_2_while_less_sequential_11_simple_rnn_2_strided_slice_1*
T0*
_output_shapes
: Б
)sequential_11/simple_rnn_2/while/IdentityIdentity)sequential_11/simple_rnn_2/while/Less:z:0*
T0
*
_output_shapes
: "_
)sequential_11_simple_rnn_2_while_identity2sequential_11/simple_rnn_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :         @: ::::: 
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
:         @:

_output_shapes
: :

_output_shapes
:
Е5
Я
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1292518

inputs+
simple_rnn_cell_4_1292441:@'
simple_rnn_cell_4_1292443:@+
simple_rnn_cell_4_1292445:@@
identityИв)simple_rnn_cell_4/StatefulPartitionedCallвwhile;
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
valueB:╤
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
value	B :@s
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
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
:         *
shrink_axis_maskы
)simple_rnn_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_4_1292441simple_rnn_cell_4_1292443simple_rnn_cell_4_1292445*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1292401n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Т
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_4_1292441simple_rnn_cell_4_1292443simple_rnn_cell_4_1292445*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1292454*
condR
while_cond_1292453*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @z
NoOpNoOp*^simple_rnn_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2V
)simple_rnn_cell_4/StatefulPartitionedCall)simple_rnn_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
л>
╝
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1292643

inputsB
0simple_rnn_cell_4_matmul_readvariableop_resource:@?
1simple_rnn_cell_4_biasadd_readvariableop_resource:@D
2simple_rnn_cell_4_matmul_1_readvariableop_resource:@@
identityИв(simple_rnn_cell_4/BiasAdd/ReadVariableOpв'simple_rnn_cell_4/MatMul/ReadVariableOpв)simple_rnn_cell_4/MatMul_1/ReadVariableOpвwhile;
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
valueB:╤
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
value	B :@s
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
:         @c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
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
:         *
shrink_axis_maskШ
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Я
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ц
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0м
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ь
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Щ
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ъ
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @k
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┌
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1292576*
condR
while_cond_1292575*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         @╧
NoOpNoOp)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╗

█
3__inference_simple_rnn_cell_4_layer_call_fn_1293787

inputs
states_0
unknown:@
	unknown_0:@
	unknown_1:@@
identity

identity_1ИвStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1292279o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :         @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         @
"
_user_specified_name
states/0
╓
я
/__inference_sequential_11_layer_call_fn_1293002

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_11_layer_call_and_return_conditional_losses_1292887o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╗

█
3__inference_simple_rnn_cell_4_layer_call_fn_1293801

inputs
states_0
unknown:@
	unknown_0:@
	unknown_1:@@
identity

identity_1ИвStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1292401o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :         @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         @
"
_user_specified_name
states/0
г
Р
J__inference_sequential_11_layer_call_and_return_conditional_losses_1292949
input_8&
simple_rnn_2_1292935:@"
simple_rnn_2_1292937:@&
simple_rnn_2_1292939:@@"
dense_17_1292943:@
dense_17_1292945:
identityИв dense_17/StatefulPartitionedCallв!dropout_4/StatefulPartitionedCallв$simple_rnn_2/StatefulPartitionedCallЬ
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallinput_8simple_rnn_2_1292935simple_rnn_2_1292937simple_rnn_2_1292939*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1292844Є
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1292718Ч
 dense_17/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_17_1292943dense_17_1292945*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1292668x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ┤
NoOpNoOp!^dense_17/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall:T P
+
_output_shapes
:         
!
_user_specified_name	input_8
╟-
╔
while_body_1293440
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0:@G
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_4_matmul_readvariableop_resource:@E
7while_simple_rnn_cell_4_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:@@Ив.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpв-while/simple_rnn_cell_4/MatMul/ReadVariableOpв/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ж
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0├
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @д
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0╛
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @к
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0к
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @м
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @w
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0 while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype0:щш╥M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: }
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:         @▀

while/NoOpNoOp/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         @: : : : : 2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:

_output_shapes
: :

_output_shapes
: 
╖c
Ї
#__inference__traced_restore_1294012
file_prefix2
 assignvariableop_dense_17_kernel:@.
 assignvariableop_1_dense_17_bias:J
8assignvariableop_2_simple_rnn_2_simple_rnn_cell_4_kernel:@T
Bassignvariableop_3_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel:@@D
6assignvariableop_4_simple_rnn_2_simple_rnn_cell_4_bias:@&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: #
assignvariableop_12_total: #
assignvariableop_13_count: <
*assignvariableop_14_adam_dense_17_kernel_m:@6
(assignvariableop_15_adam_dense_17_bias_m:R
@assignvariableop_16_adam_simple_rnn_2_simple_rnn_cell_4_kernel_m:@\
Jassignvariableop_17_adam_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_m:@@L
>assignvariableop_18_adam_simple_rnn_2_simple_rnn_cell_4_bias_m:@<
*assignvariableop_19_adam_dense_17_kernel_v:@6
(assignvariableop_20_adam_dense_17_bias_v:R
@assignvariableop_21_adam_simple_rnn_2_simple_rnn_cell_4_kernel_v:@\
Jassignvariableop_22_adam_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_v:@@L
>assignvariableop_23_adam_simple_rnn_2_simple_rnn_cell_4_bias_v:@
identity_25ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Ф
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*║
value░BнB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHв
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B Ы
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOpAssignVariableOp assignvariableop_dense_17_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_17_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_2AssignVariableOp8assignvariableop_2_simple_rnn_2_simple_rnn_cell_4_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_3AssignVariableOpBassignvariableop_3_simple_rnn_2_simple_rnn_cell_4_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_4AssignVariableOp6assignvariableop_4_simple_rnn_2_simple_rnn_cell_4_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_dense_17_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_dense_17_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_16AssignVariableOp@assignvariableop_16_adam_simple_rnn_2_simple_rnn_cell_4_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_17AssignVariableOpJassignvariableop_17_adam_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_18AssignVariableOp>assignvariableop_18_adam_simple_rnn_2_simple_rnn_cell_4_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_17_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_17_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_simple_rnn_2_simple_rnn_cell_4_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_22AssignVariableOpJassignvariableop_22_adam_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_simple_rnn_2_simple_rnn_cell_4_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ▀
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: ╠
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
╚	
Ў
E__inference_dense_17_layer_call_and_return_conditional_losses_1293773

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╟-
╔
while_body_1293330
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0:@G
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_4_matmul_readvariableop_resource:@E
7while_simple_rnn_cell_4_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:@@Ив.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpв-while/simple_rnn_cell_4/MatMul/ReadVariableOpв/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ж
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0├
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @д
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0╛
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @к
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0к
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @м
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @w
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0 while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype0:щш╥M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: }
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:         @▀

while/NoOpNoOp/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         @: : : : : 2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:

_output_shapes
: :

_output_shapes
: 
ЦS
╬
J__inference_sequential_11_layer_call_and_return_conditional_losses_1293119

inputsO
=simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resource:@L
>simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resource:@Q
?simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resource:@@9
'dense_17_matmul_readvariableop_resource:@6
(dense_17_biasadd_readvariableop_resource:
identityИвdense_17/BiasAdd/ReadVariableOpвdense_17/MatMul/ReadVariableOpв5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOpв4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOpв6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOpвsimple_rnn_2/whileH
simple_rnn_2/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
simple_rnn_2/strided_sliceStridedSlicesimple_rnn_2/Shape:output:0)simple_rnn_2/strided_slice/stack:output:0+simple_rnn_2/strided_slice/stack_1:output:0+simple_rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@Ъ
simple_rnn_2/zeros/packedPack#simple_rnn_2/strided_slice:output:0$simple_rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    У
simple_rnn_2/zerosFill"simple_rnn_2/zeros/packed:output:0!simple_rnn_2/zeros/Const:output:0*
T0*'
_output_shapes
:         @p
simple_rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          З
simple_rnn_2/transpose	Transposeinputs$simple_rnn_2/transpose/perm:output:0*
T0*+
_output_shapes
:         ^
simple_rnn_2/Shape_1Shapesimple_rnn_2/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
simple_rnn_2/strided_slice_1StridedSlicesimple_rnn_2/Shape_1:output:0+simple_rnn_2/strided_slice_1/stack:output:0-simple_rnn_2/strided_slice_1/stack_1:output:0-simple_rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         █
simple_rnn_2/TensorArrayV2TensorListReserve1simple_rnn_2/TensorArrayV2/element_shape:output:0%simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥У
Bsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       З
4simple_rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_2/transpose:y:0Ksimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥l
"simple_rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
simple_rnn_2/strided_slice_2StridedSlicesimple_rnn_2/transpose:y:0+simple_rnn_2/strided_slice_2/stack:output:0-simple_rnn_2/strided_slice_2/stack_1:output:0-simple_rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask▓
4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp=simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0╞
%simple_rnn_2/simple_rnn_cell_4/MatMulMatMul%simple_rnn_2/strided_slice_2:output:0<simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @░
5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╙
&simple_rnn_2/simple_rnn_cell_4/BiasAddBiasAdd/simple_rnn_2/simple_rnn_cell_4/MatMul:product:0=simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╢
6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0└
'simple_rnn_2/simple_rnn_cell_4/MatMul_1MatMulsimple_rnn_2/zeros:output:0>simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @┴
"simple_rnn_2/simple_rnn_cell_4/addAddV2/simple_rnn_2/simple_rnn_cell_4/BiasAdd:output:01simple_rnn_2/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @Е
#simple_rnn_2/simple_rnn_cell_4/TanhTanh&simple_rnn_2/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @{
*simple_rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   k
)simple_rnn_2/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ь
simple_rnn_2/TensorArrayV2_1TensorListReserve3simple_rnn_2/TensorArrayV2_1/element_shape:output:02simple_rnn_2/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥S
simple_rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         a
simple_rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Г
simple_rnn_2/whileWhile(simple_rnn_2/while/loop_counter:output:0.simple_rnn_2/while/maximum_iterations:output:0simple_rnn_2/time:output:0%simple_rnn_2/TensorArrayV2_1:handle:0simple_rnn_2/zeros:output:0%simple_rnn_2/strided_slice_1:output:0Dsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resource>simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resource?simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
simple_rnn_2_while_body_1293045*+
cond#R!
simple_rnn_2_while_cond_1293044*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations О
=simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ¤
/simple_rnn_2/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_2/while:output:3Fsimple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsu
"simple_rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         n
$simple_rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╚
simple_rnn_2/strided_slice_3StridedSlice8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_2/strided_slice_3/stack:output:0-simple_rnn_2/strided_slice_3/stack_1:output:0-simple_rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskr
simple_rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╜
simple_rnn_2/transpose_1	Transpose8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:         @w
dropout_4/IdentityIdentity%simple_rnn_2/strided_slice_3:output:0*
T0*'
_output_shapes
:         @Ж
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Р
dense_17/MatMulMatMuldropout_4/Identity:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ╞
NoOpNoOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp6^simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp5^simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp7^simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp^simple_rnn_2/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2n
5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp2l
4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp2p
6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp2(
simple_rnn_2/whilesimple_rnn_2/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
┐

ж
simple_rnn_2_while_cond_12930446
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_28
4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1O
Ksimple_rnn_2_while_simple_rnn_2_while_cond_1293044___redundant_placeholder0O
Ksimple_rnn_2_while_simple_rnn_2_while_cond_1293044___redundant_placeholder1O
Ksimple_rnn_2_while_simple_rnn_2_while_cond_1293044___redundant_placeholder2O
Ksimple_rnn_2_while_simple_rnn_2_while_cond_1293044___redundant_placeholder3
simple_rnn_2_while_identity
Ц
simple_rnn_2/while/LessLesssimple_rnn_2_while_placeholder4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_2/while/IdentityIdentitysimple_rnn_2/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :         @: ::::: 
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
:         @:

_output_shapes
: :

_output_shapes
:
▀
п
while_cond_1293329
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1293329___redundant_placeholder05
1while_while_cond_1293329___redundant_placeholder15
1while_while_cond_1293329___redundant_placeholder25
1while_while_cond_1293329___redundant_placeholder3
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
-: : : : :         @: ::::: 
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
:         @:

_output_shapes
: :

_output_shapes
:
Ї	
e
F__inference_dropout_4_layer_call_and_return_conditional_losses_1292718

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Ї	
e
F__inference_dropout_4_layer_call_and_return_conditional_losses_1293754

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
▀
п
while_cond_1292575
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1292575___redundant_placeholder05
1while_while_cond_1292575___redundant_placeholder15
1while_while_cond_1292575___redundant_placeholder25
1while_while_cond_1292575___redundant_placeholder3
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
-: : : : :         @: ::::: 
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
:         @:

_output_shapes
: :

_output_shapes
:
шb
┴
"__inference__wrapped_model_1292231
input_8]
Ksequential_11_simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resource:@Z
Lsequential_11_simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resource:@_
Msequential_11_simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resource:@@G
5sequential_11_dense_17_matmul_readvariableop_resource:@D
6sequential_11_dense_17_biasadd_readvariableop_resource:
identityИв-sequential_11/dense_17/BiasAdd/ReadVariableOpв,sequential_11/dense_17/MatMul/ReadVariableOpвCsequential_11/simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOpвBsequential_11/simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOpвDsequential_11/simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOpв sequential_11/simple_rnn_2/whileW
 sequential_11/simple_rnn_2/ShapeShapeinput_8*
T0*
_output_shapes
:x
.sequential_11/simple_rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential_11/simple_rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential_11/simple_rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╪
(sequential_11/simple_rnn_2/strided_sliceStridedSlice)sequential_11/simple_rnn_2/Shape:output:07sequential_11/simple_rnn_2/strided_slice/stack:output:09sequential_11/simple_rnn_2/strided_slice/stack_1:output:09sequential_11/simple_rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_11/simple_rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@─
'sequential_11/simple_rnn_2/zeros/packedPack1sequential_11/simple_rnn_2/strided_slice:output:02sequential_11/simple_rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&sequential_11/simple_rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╜
 sequential_11/simple_rnn_2/zerosFill0sequential_11/simple_rnn_2/zeros/packed:output:0/sequential_11/simple_rnn_2/zeros/Const:output:0*
T0*'
_output_shapes
:         @~
)sequential_11/simple_rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          д
$sequential_11/simple_rnn_2/transpose	Transposeinput_82sequential_11/simple_rnn_2/transpose/perm:output:0*
T0*+
_output_shapes
:         z
"sequential_11/simple_rnn_2/Shape_1Shape(sequential_11/simple_rnn_2/transpose:y:0*
T0*
_output_shapes
:z
0sequential_11/simple_rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_11/simple_rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_11/simple_rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
*sequential_11/simple_rnn_2/strided_slice_1StridedSlice+sequential_11/simple_rnn_2/Shape_1:output:09sequential_11/simple_rnn_2/strided_slice_1/stack:output:0;sequential_11/simple_rnn_2/strided_slice_1/stack_1:output:0;sequential_11/simple_rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
6sequential_11/simple_rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         Е
(sequential_11/simple_rnn_2/TensorArrayV2TensorListReserve?sequential_11/simple_rnn_2/TensorArrayV2/element_shape:output:03sequential_11/simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥б
Psequential_11/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ▒
Bsequential_11/simple_rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_11/simple_rnn_2/transpose:y:0Ysequential_11/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥z
0sequential_11/simple_rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_11/simple_rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_11/simple_rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
*sequential_11/simple_rnn_2/strided_slice_2StridedSlice(sequential_11/simple_rnn_2/transpose:y:09sequential_11/simple_rnn_2/strided_slice_2/stack:output:0;sequential_11/simple_rnn_2/strided_slice_2/stack_1:output:0;sequential_11/simple_rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask╬
Bsequential_11/simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOpKsequential_11_simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ё
3sequential_11/simple_rnn_2/simple_rnn_cell_4/MatMulMatMul3sequential_11/simple_rnn_2/strided_slice_2:output:0Jsequential_11/simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╠
Csequential_11/simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOpLsequential_11_simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¤
4sequential_11/simple_rnn_2/simple_rnn_cell_4/BiasAddBiasAdd=sequential_11/simple_rnn_2/simple_rnn_cell_4/MatMul:product:0Ksequential_11/simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╥
Dsequential_11/simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOpMsequential_11_simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0ъ
5sequential_11/simple_rnn_2/simple_rnn_cell_4/MatMul_1MatMul)sequential_11/simple_rnn_2/zeros:output:0Lsequential_11/simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ы
0sequential_11/simple_rnn_2/simple_rnn_cell_4/addAddV2=sequential_11/simple_rnn_2/simple_rnn_cell_4/BiasAdd:output:0?sequential_11/simple_rnn_2/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @б
1sequential_11/simple_rnn_2/simple_rnn_cell_4/TanhTanh4sequential_11/simple_rnn_2/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @Й
8sequential_11/simple_rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   y
7sequential_11/simple_rnn_2/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ц
*sequential_11/simple_rnn_2/TensorArrayV2_1TensorListReserveAsequential_11/simple_rnn_2/TensorArrayV2_1/element_shape:output:0@sequential_11/simple_rnn_2/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥a
sequential_11/simple_rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3sequential_11/simple_rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         o
-sequential_11/simple_rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ╣
 sequential_11/simple_rnn_2/whileWhile6sequential_11/simple_rnn_2/while/loop_counter:output:0<sequential_11/simple_rnn_2/while/maximum_iterations:output:0(sequential_11/simple_rnn_2/time:output:03sequential_11/simple_rnn_2/TensorArrayV2_1:handle:0)sequential_11/simple_rnn_2/zeros:output:03sequential_11/simple_rnn_2/strided_slice_1:output:0Rsequential_11/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ksequential_11_simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resourceLsequential_11_simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resourceMsequential_11_simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *9
body1R/
-sequential_11_simple_rnn_2_while_body_1292157*9
cond1R/
-sequential_11_simple_rnn_2_while_cond_1292156*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations Ь
Ksequential_11/simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   з
=sequential_11/simple_rnn_2/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_11/simple_rnn_2/while:output:3Tsequential_11/simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsГ
0sequential_11/simple_rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         |
2sequential_11/simple_rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2sequential_11/simple_rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
*sequential_11/simple_rnn_2/strided_slice_3StridedSliceFsequential_11/simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:09sequential_11/simple_rnn_2/strided_slice_3/stack:output:0;sequential_11/simple_rnn_2/strided_slice_3/stack_1:output:0;sequential_11/simple_rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskА
+sequential_11/simple_rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ч
&sequential_11/simple_rnn_2/transpose_1	TransposeFsequential_11/simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:04sequential_11/simple_rnn_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:         @У
 sequential_11/dropout_4/IdentityIdentity3sequential_11/simple_rnn_2/strided_slice_3:output:0*
T0*'
_output_shapes
:         @в
,sequential_11/dense_17/MatMul/ReadVariableOpReadVariableOp5sequential_11_dense_17_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0║
sequential_11/dense_17/MatMulMatMul)sequential_11/dropout_4/Identity:output:04sequential_11/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
-sequential_11/dense_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╗
sequential_11/dense_17/BiasAddBiasAdd'sequential_11/dense_17/MatMul:product:05sequential_11/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         v
IdentityIdentity'sequential_11/dense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         Ъ
NoOpNoOp.^sequential_11/dense_17/BiasAdd/ReadVariableOp-^sequential_11/dense_17/MatMul/ReadVariableOpD^sequential_11/simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOpC^sequential_11/simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOpE^sequential_11/simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp!^sequential_11/simple_rnn_2/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2^
-sequential_11/dense_17/BiasAdd/ReadVariableOp-sequential_11/dense_17/BiasAdd/ReadVariableOp2\
,sequential_11/dense_17/MatMul/ReadVariableOp,sequential_11/dense_17/MatMul/ReadVariableOp2К
Csequential_11/simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOpCsequential_11/simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp2И
Bsequential_11/simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOpBsequential_11/simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp2М
Dsequential_11/simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOpDsequential_11/simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp2D
 sequential_11/simple_rnn_2/while sequential_11/simple_rnn_2/while:T P
+
_output_shapes
:         
!
_user_specified_name	input_8
щ9
╧
simple_rnn_2_while_body_12931626
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_25
1simple_rnn_2_while_simple_rnn_2_strided_slice_1_0q
msimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0:@T
Fsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:@Y
Gsimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:@@
simple_rnn_2_while_identity!
simple_rnn_2_while_identity_1!
simple_rnn_2_while_identity_2!
simple_rnn_2_while_identity_3!
simple_rnn_2_while_identity_43
/simple_rnn_2_while_simple_rnn_2_strided_slice_1o
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource:@R
Dsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource:@W
Esimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource:@@Ив;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpв:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOpв<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpХ
Dsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ч
6simple_rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_2_while_placeholderMsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0└
:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0ъ
+simple_rnn_2/while/simple_rnn_cell_4/MatMulMatMul=simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╛
;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0х
,simple_rnn_2/while/simple_rnn_cell_4/BiasAddBiasAdd5simple_rnn_2/while/simple_rnn_cell_4/MatMul:product:0Csimple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @─
<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0╤
-simple_rnn_2/while/simple_rnn_cell_4/MatMul_1MatMul simple_rnn_2_while_placeholder_2Dsimple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╙
(simple_rnn_2/while/simple_rnn_cell_4/addAddV25simple_rnn_2/while/simple_rnn_cell_4/BiasAdd:output:07simple_rnn_2/while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @С
)simple_rnn_2/while/simple_rnn_cell_4/TanhTanh,simple_rnn_2/while/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @
=simple_rnn_2/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : е
7simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_2_while_placeholder_1Fsimple_rnn_2/while/TensorArrayV2Write/TensorListSetItem/index:output:0-simple_rnn_2/while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype0:щш╥Z
simple_rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Г
simple_rnn_2/while/addAddV2simple_rnn_2_while_placeholder!simple_rnn_2/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ы
simple_rnn_2/while/add_1AddV22simple_rnn_2_while_simple_rnn_2_while_loop_counter#simple_rnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: А
simple_rnn_2/while/IdentityIdentitysimple_rnn_2/while/add_1:z:0^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: Ю
simple_rnn_2/while/Identity_1Identity8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: А
simple_rnn_2/while/Identity_2Identitysimple_rnn_2/while/add:z:0^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: н
simple_rnn_2/while/Identity_3IdentityGsimple_rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: д
simple_rnn_2/while/Identity_4Identity-simple_rnn_2/while/simple_rnn_cell_4/Tanh:y:0^simple_rnn_2/while/NoOp*
T0*'
_output_shapes
:         @У
simple_rnn_2/while/NoOpNoOp<^simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0"G
simple_rnn_2_while_identity_1&simple_rnn_2/while/Identity_1:output:0"G
simple_rnn_2_while_identity_2&simple_rnn_2/while/Identity_2:output:0"G
simple_rnn_2_while_identity_3&simple_rnn_2/while/Identity_3:output:0"G
simple_rnn_2_while_identity_4&simple_rnn_2/while/Identity_4:output:0"d
/simple_rnn_2_while_simple_rnn_2_strided_slice_11simple_rnn_2_while_simple_rnn_2_strided_slice_1_0"О
Dsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resourceFsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"Р
Esimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resourceGsimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"М
Csimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resourceEsimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0"▄
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensormsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         @: : : : : 2z
;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2x
:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp2|
<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:

_output_shapes
: :

_output_shapes
: 
┐

ж
simple_rnn_2_while_cond_12931616
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_28
4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1O
Ksimple_rnn_2_while_simple_rnn_2_while_cond_1293161___redundant_placeholder0O
Ksimple_rnn_2_while_simple_rnn_2_while_cond_1293161___redundant_placeholder1O
Ksimple_rnn_2_while_simple_rnn_2_while_cond_1293161___redundant_placeholder2O
Ksimple_rnn_2_while_simple_rnn_2_while_cond_1293161___redundant_placeholder3
simple_rnn_2_while_identity
Ц
simple_rnn_2/while/LessLesssimple_rnn_2_while_placeholder4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_2/while/IdentityIdentitysimple_rnn_2/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :         @: ::::: 
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
:         @:

_output_shapes
: :

_output_shapes
:
╓
я
/__inference_sequential_11_layer_call_fn_1292987

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_11_layer_call_and_return_conditional_losses_1292675o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
▀
п
while_cond_1292776
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1292776___redundant_placeholder05
1while_while_cond_1292776___redundant_placeholder15
1while_while_cond_1292776___redundant_placeholder25
1while_while_cond_1292776___redundant_placeholder3
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
-: : : : :         @: ::::: 
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
:         @:

_output_shapes
: :

_output_shapes
:
─
Ч
*__inference_dense_17_layer_call_fn_1293763

inputs
unknown:@
	unknown_0:
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1292668o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
фZ
╬
J__inference_sequential_11_layer_call_and_return_conditional_losses_1293243

inputsO
=simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resource:@L
>simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resource:@Q
?simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resource:@@9
'dense_17_matmul_readvariableop_resource:@6
(dense_17_biasadd_readvariableop_resource:
identityИвdense_17/BiasAdd/ReadVariableOpвdense_17/MatMul/ReadVariableOpв5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOpв4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOpв6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOpвsimple_rnn_2/whileH
simple_rnn_2/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
simple_rnn_2/strided_sliceStridedSlicesimple_rnn_2/Shape:output:0)simple_rnn_2/strided_slice/stack:output:0+simple_rnn_2/strided_slice/stack_1:output:0+simple_rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@Ъ
simple_rnn_2/zeros/packedPack#simple_rnn_2/strided_slice:output:0$simple_rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    У
simple_rnn_2/zerosFill"simple_rnn_2/zeros/packed:output:0!simple_rnn_2/zeros/Const:output:0*
T0*'
_output_shapes
:         @p
simple_rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          З
simple_rnn_2/transpose	Transposeinputs$simple_rnn_2/transpose/perm:output:0*
T0*+
_output_shapes
:         ^
simple_rnn_2/Shape_1Shapesimple_rnn_2/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
simple_rnn_2/strided_slice_1StridedSlicesimple_rnn_2/Shape_1:output:0+simple_rnn_2/strided_slice_1/stack:output:0-simple_rnn_2/strided_slice_1/stack_1:output:0-simple_rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         █
simple_rnn_2/TensorArrayV2TensorListReserve1simple_rnn_2/TensorArrayV2/element_shape:output:0%simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥У
Bsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       З
4simple_rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_2/transpose:y:0Ksimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥l
"simple_rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
simple_rnn_2/strided_slice_2StridedSlicesimple_rnn_2/transpose:y:0+simple_rnn_2/strided_slice_2/stack:output:0-simple_rnn_2/strided_slice_2/stack_1:output:0-simple_rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask▓
4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp=simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0╞
%simple_rnn_2/simple_rnn_cell_4/MatMulMatMul%simple_rnn_2/strided_slice_2:output:0<simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @░
5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╙
&simple_rnn_2/simple_rnn_cell_4/BiasAddBiasAdd/simple_rnn_2/simple_rnn_cell_4/MatMul:product:0=simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╢
6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0└
'simple_rnn_2/simple_rnn_cell_4/MatMul_1MatMulsimple_rnn_2/zeros:output:0>simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @┴
"simple_rnn_2/simple_rnn_cell_4/addAddV2/simple_rnn_2/simple_rnn_cell_4/BiasAdd:output:01simple_rnn_2/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @Е
#simple_rnn_2/simple_rnn_cell_4/TanhTanh&simple_rnn_2/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @{
*simple_rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   k
)simple_rnn_2/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ь
simple_rnn_2/TensorArrayV2_1TensorListReserve3simple_rnn_2/TensorArrayV2_1/element_shape:output:02simple_rnn_2/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥S
simple_rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         a
simple_rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Г
simple_rnn_2/whileWhile(simple_rnn_2/while/loop_counter:output:0.simple_rnn_2/while/maximum_iterations:output:0simple_rnn_2/time:output:0%simple_rnn_2/TensorArrayV2_1:handle:0simple_rnn_2/zeros:output:0%simple_rnn_2/strided_slice_1:output:0Dsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resource>simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resource?simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
simple_rnn_2_while_body_1293162*+
cond#R!
simple_rnn_2_while_cond_1293161*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations О
=simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ¤
/simple_rnn_2/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_2/while:output:3Fsimple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype0*
num_elementsu
"simple_rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         n
$simple_rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╚
simple_rnn_2/strided_slice_3StridedSlice8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_2/strided_slice_3/stack:output:0-simple_rnn_2/strided_slice_3/stack_1:output:0-simple_rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_maskr
simple_rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╜
simple_rnn_2/transpose_1	Transpose8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:         @\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?Ч
dropout_4/dropout/MulMul%simple_rnn_2/strided_slice_3:output:0 dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:         @l
dropout_4/dropout/ShapeShape%simple_rnn_2/strided_slice_3:output:0*
T0*
_output_shapes
:а
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>─
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @Г
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @З
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:         @Ж
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Р
dense_17/MatMulMatMuldropout_4/dropout/Mul_1:z:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ╞
NoOpNoOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp6^simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp5^simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp7^simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp^simple_rnn_2/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2n
5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp2l
4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp2p
6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp2(
simple_rnn_2/whilesimple_rnn_2/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╚	
Ў
E__inference_dense_17_layer_call_and_return_conditional_losses_1292668

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
г"
╪
while_body_1292454
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
!while_simple_rnn_cell_4_1292476_0:@/
!while_simple_rnn_cell_4_1292478_0:@3
!while_simple_rnn_cell_4_1292480_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
while_simple_rnn_cell_4_1292476:@-
while_simple_rnn_cell_4_1292478:@1
while_simple_rnn_cell_4_1292480:@@Ив/while/simple_rnn_cell_4/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ж
/while/simple_rnn_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_4_1292476_0!while_simple_rnn_cell_4_1292478_0!while_simple_rnn_cell_4_1292480_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1292401r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Й
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:08while/simple_rnn_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Х
while/Identity_4Identity8while/simple_rnn_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         @~

while/NoOpNoOp0^while/simple_rnn_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_4_1292476!while_simple_rnn_cell_4_1292476_0"D
while_simple_rnn_cell_4_1292478!while_simple_rnn_cell_4_1292478_0"D
while_simple_rnn_cell_4_1292480!while_simple_rnn_cell_4_1292480_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         @: : : : : 2b
/while/simple_rnn_cell_4/StatefulPartitionedCall/while/simple_rnn_cell_4/StatefulPartitionedCall: 
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
:         @:

_output_shapes
: :

_output_shapes
: 
╟-
╔
while_body_1292777
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0:@G
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_4_matmul_readvariableop_resource:@E
7while_simple_rnn_cell_4_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:@@Ив.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpв-while/simple_rnn_cell_4/MatMul/ReadVariableOpв/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ж
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0├
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @д
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0╛
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @к
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0к
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @м
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @w
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0 while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype0:щш╥M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: }
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:         @▀

while/NoOpNoOp/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         @: : : : : 2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:

_output_shapes
: :

_output_shapes
: 
є
d
+__inference_dropout_4_layer_call_fn_1293737

inputs
identityИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1292718o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
┘
d
F__inference_dropout_4_layer_call_and_return_conditional_losses_1292656

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         @[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
А
╕
.__inference_simple_rnn_2_layer_call_fn_1293287

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1292844o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
°
ы
J__inference_sequential_11_layer_call_and_return_conditional_losses_1292675

inputs&
simple_rnn_2_1292644:@"
simple_rnn_2_1292646:@&
simple_rnn_2_1292648:@@"
dense_17_1292669:@
dense_17_1292671:
identityИв dense_17/StatefulPartitionedCallв$simple_rnn_2/StatefulPartitionedCallЫ
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_2_1292644simple_rnn_2_1292646simple_rnn_2_1292648*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1292643т
dropout_4/PartitionedCallPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1292656П
 dense_17/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_17_1292669dense_17_1292671*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1292668x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Р
NoOpNoOp!^dense_17/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╟-
╔
while_body_1293660
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0:@G
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_4_matmul_readvariableop_resource:@E
7while_simple_rnn_cell_4_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:@@Ив.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpв-while/simple_rnn_cell_4/MatMul/ReadVariableOpв/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ж
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0├
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @д
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0╛
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @к
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0к
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @м
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:         @w
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:         @r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0 while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype0:щш╥M
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
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: }
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:         @▀

while/NoOpNoOp/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :         @: : : : : 2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:

_output_shapes
: :

_output_shapes
: 
■
щ
N__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1292279

inputs

states0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:         @G
TanhTanhadd:z:0*
T0*'
_output_shapes
:         @W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:         @Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:         @С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :         @: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_namestates
■
щ
N__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1292401

inputs

states0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:         @G
TanhTanhadd:z:0*
T0*'
_output_shapes
:         @W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:         @Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:         @С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:         :         @: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_namestates"╡	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*п
serving_defaultЫ
?
input_84
serving_default_input_8:0         <
dense_170
StatefulPartitionedCall:0         tensorflow/serving/predict:█▒
┴
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
├
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec"
_tf_keras_rnn_layer
╝
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
╗
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias"
_tf_keras_layer
C
$0
%1
&2
"3
#4"
trackable_list_wrapper
C
$0
%1
&2
"3
#4"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
ё
,trace_0
-trace_1
.trace_2
/trace_32Ж
/__inference_sequential_11_layer_call_fn_1292688
/__inference_sequential_11_layer_call_fn_1292987
/__inference_sequential_11_layer_call_fn_1293002
/__inference_sequential_11_layer_call_fn_1292915┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z,trace_0z-trace_1z.trace_2z/trace_3
▌
0trace_0
1trace_1
2trace_2
3trace_32Є
J__inference_sequential_11_layer_call_and_return_conditional_losses_1293119
J__inference_sequential_11_layer_call_and_return_conditional_losses_1293243
J__inference_sequential_11_layer_call_and_return_conditional_losses_1292932
J__inference_sequential_11_layer_call_and_return_conditional_losses_1292949┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z0trace_0z1trace_1z2trace_2z3trace_3
═B╩
"__inference__wrapped_model_1292231input_8"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
н
4iter

5beta_1

6beta_2
	7decay
8learning_rate"mr#ms$mt%mu&mv"vw#vx$vy%vz&v{"
	optimizer
,
9serving_default"
signature_map
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
 "
trackable_list_wrapper
╣

:states
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
В
@trace_0
Atrace_1
Btrace_2
Ctrace_32Ч
.__inference_simple_rnn_2_layer_call_fn_1293254
.__inference_simple_rnn_2_layer_call_fn_1293265
.__inference_simple_rnn_2_layer_call_fn_1293276
.__inference_simple_rnn_2_layer_call_fn_1293287╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z@trace_0zAtrace_1zBtrace_2zCtrace_3
ю
Dtrace_0
Etrace_1
Ftrace_2
Gtrace_32Г
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1293397
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1293507
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1293617
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1293727╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zDtrace_0zEtrace_1zFtrace_2zGtrace_3
ш
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
N_random_generator

$kernel
%recurrent_kernel
&bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╟
Ttrace_0
Utrace_12Р
+__inference_dropout_4_layer_call_fn_1293732
+__inference_dropout_4_layer_call_fn_1293737│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zTtrace_0zUtrace_1
¤
Vtrace_0
Wtrace_12╞
F__inference_dropout_4_layer_call_and_return_conditional_losses_1293742
F__inference_dropout_4_layer_call_and_return_conditional_losses_1293754│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zVtrace_0zWtrace_1
"
_generic_user_object
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
ю
]trace_02╤
*__inference_dense_17_layer_call_fn_1293763в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z]trace_0
Й
^trace_02ь
E__inference_dense_17_layer_call_and_return_conditional_losses_1293773в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z^trace_0
!:@2dense_17/kernel
:2dense_17/bias
7:5@2%simple_rnn_2/simple_rnn_cell_4/kernel
A:?@@2/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel
1:/@2#simple_rnn_2/simple_rnn_cell_4/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
БB■
/__inference_sequential_11_layer_call_fn_1292688input_8"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
/__inference_sequential_11_layer_call_fn_1292987inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
/__inference_sequential_11_layer_call_fn_1293002inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
БB■
/__inference_sequential_11_layer_call_fn_1292915input_8"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЫBШ
J__inference_sequential_11_layer_call_and_return_conditional_losses_1293119inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЫBШ
J__inference_sequential_11_layer_call_and_return_conditional_losses_1293243inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЬBЩ
J__inference_sequential_11_layer_call_and_return_conditional_losses_1292932input_8"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЬBЩ
J__inference_sequential_11_layer_call_and_return_conditional_losses_1292949input_8"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
╠B╔
%__inference_signature_wrapper_1292972input_8"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЦBУ
.__inference_simple_rnn_2_layer_call_fn_1293254inputs/0"╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЦBУ
.__inference_simple_rnn_2_layer_call_fn_1293265inputs/0"╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
.__inference_simple_rnn_2_layer_call_fn_1293276inputs"╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
.__inference_simple_rnn_2_layer_call_fn_1293287inputs"╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▒Bо
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1293397inputs/0"╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▒Bо
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1293507inputs/0"╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
пBм
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1293617inputs"╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
пBм
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1293727inputs"╘
╦▓╟
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
 "
trackable_list_wrapper
н
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
с
ftrace_0
gtrace_12к
3__inference_simple_rnn_cell_4_layer_call_fn_1293787
3__inference_simple_rnn_cell_4_layer_call_fn_1293801╜
┤▓░
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zftrace_0zgtrace_1
Ч
htrace_0
itrace_12р
N__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1293818
N__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1293835╜
┤▓░
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zhtrace_0zitrace_1
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
ЁBэ
+__inference_dropout_4_layer_call_fn_1293732inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЁBэ
+__inference_dropout_4_layer_call_fn_1293737inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЛBИ
F__inference_dropout_4_layer_call_and_return_conditional_losses_1293742inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЛBИ
F__inference_dropout_4_layer_call_and_return_conditional_losses_1293754inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
▐B█
*__inference_dense_17_layer_call_fn_1293763inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_dense_17_layer_call_and_return_conditional_losses_1293773inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
N
j	variables
k	keras_api
	ltotal
	mcount"
_tf_keras_metric
N
n	variables
o	keras_api
	ptotal
	qcount"
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
МBЙ
3__inference_simple_rnn_cell_4_layer_call_fn_1293787inputsstates/0"╜
┤▓░
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
МBЙ
3__inference_simple_rnn_cell_4_layer_call_fn_1293801inputsstates/0"╜
┤▓░
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
зBд
N__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1293818inputsstates/0"╜
┤▓░
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
зBд
N__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1293835inputsstates/0"╜
┤▓░
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
l0
m1"
trackable_list_wrapper
-
j	variables"
_generic_user_object
:  (2total
:  (2count
.
p0
q1"
trackable_list_wrapper
-
n	variables"
_generic_user_object
:  (2total
:  (2count
&:$@2Adam/dense_17/kernel/m
 :2Adam/dense_17/bias/m
<::@2,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/m
F:D@@26Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/m
6:4@2*Adam/simple_rnn_2/simple_rnn_cell_4/bias/m
&:$@2Adam/dense_17/kernel/v
 :2Adam/dense_17/bias/v
<::@2,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/v
F:D@@26Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/v
6:4@2*Adam/simple_rnn_2/simple_rnn_cell_4/bias/vШ
"__inference__wrapped_model_1292231r$&%"#4в1
*в'
%К"
input_8         
к "3к0
.
dense_17"К
dense_17         е
E__inference_dense_17_layer_call_and_return_conditional_losses_1293773\"#/в,
%в"
 К
inputs         @
к "%в"
К
0         
Ъ }
*__inference_dense_17_layer_call_fn_1293763O"#/в,
%в"
 К
inputs         @
к "К         ж
F__inference_dropout_4_layer_call_and_return_conditional_losses_1293742\3в0
)в&
 К
inputs         @
p 
к "%в"
К
0         @
Ъ ж
F__inference_dropout_4_layer_call_and_return_conditional_losses_1293754\3в0
)в&
 К
inputs         @
p
к "%в"
К
0         @
Ъ ~
+__inference_dropout_4_layer_call_fn_1293732O3в0
)в&
 К
inputs         @
p 
к "К         @~
+__inference_dropout_4_layer_call_fn_1293737O3в0
)в&
 К
inputs         @
p
к "К         @║
J__inference_sequential_11_layer_call_and_return_conditional_losses_1292932l$&%"#<в9
2в/
%К"
input_8         
p 

 
к "%в"
К
0         
Ъ ║
J__inference_sequential_11_layer_call_and_return_conditional_losses_1292949l$&%"#<в9
2в/
%К"
input_8         
p

 
к "%в"
К
0         
Ъ ╣
J__inference_sequential_11_layer_call_and_return_conditional_losses_1293119k$&%"#;в8
1в.
$К!
inputs         
p 

 
к "%в"
К
0         
Ъ ╣
J__inference_sequential_11_layer_call_and_return_conditional_losses_1293243k$&%"#;в8
1в.
$К!
inputs         
p

 
к "%в"
К
0         
Ъ Т
/__inference_sequential_11_layer_call_fn_1292688_$&%"#<в9
2в/
%К"
input_8         
p 

 
к "К         Т
/__inference_sequential_11_layer_call_fn_1292915_$&%"#<в9
2в/
%К"
input_8         
p

 
к "К         С
/__inference_sequential_11_layer_call_fn_1292987^$&%"#;в8
1в.
$К!
inputs         
p 

 
к "К         С
/__inference_sequential_11_layer_call_fn_1293002^$&%"#;в8
1в.
$К!
inputs         
p

 
к "К         ж
%__inference_signature_wrapper_1292972}$&%"#?в<
в 
5к2
0
input_8%К"
input_8         "3к0
.
dense_17"К
dense_17         ╩
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1293397}$&%OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p 

 
к "%в"
К
0         @
Ъ ╩
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1293507}$&%OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p

 
к "%в"
К
0         @
Ъ ║
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1293617m$&%?в<
5в2
$К!
inputs         

 
p 

 
к "%в"
К
0         @
Ъ ║
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1293727m$&%?в<
5в2
$К!
inputs         

 
p

 
к "%в"
К
0         @
Ъ в
.__inference_simple_rnn_2_layer_call_fn_1293254p$&%OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p 

 
к "К         @в
.__inference_simple_rnn_2_layer_call_fn_1293265p$&%OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p

 
к "К         @Т
.__inference_simple_rnn_2_layer_call_fn_1293276`$&%?в<
5в2
$К!
inputs         

 
p 

 
к "К         @Т
.__inference_simple_rnn_2_layer_call_fn_1293287`$&%?в<
5в2
$К!
inputs         

 
p

 
к "К         @К
N__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1293818╖$&%\вY
RвO
 К
inputs         
'в$
"К
states/0         @
p 
к "RвO
HвE
К
0/0         @
$Ъ!
К
0/1/0         @
Ъ К
N__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1293835╖$&%\вY
RвO
 К
inputs         
'в$
"К
states/0         @
p
к "RвO
HвE
К
0/0         @
$Ъ!
К
0/1/0         @
Ъ с
3__inference_simple_rnn_cell_4_layer_call_fn_1293787й$&%\вY
RвO
 К
inputs         
'в$
"К
states/0         @
p 
к "DвA
К
0         @
"Ъ
К
1/0         @с
3__inference_simple_rnn_cell_4_layer_call_fn_1293801й$&%\вY
RвO
 К
inputs         
'в$
"К
states/0         @
p
к "DвA
К
0         @
"Ъ
К
1/0         @