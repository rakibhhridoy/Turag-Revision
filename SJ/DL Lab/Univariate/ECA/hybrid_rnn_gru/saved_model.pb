��4
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
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
executor_typestring ��
�
StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
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
�"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��2
�
Adam/gru_4/gru_cell_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*-
shared_nameAdam/gru_4/gru_cell_8/bias/v
�
0Adam/gru_4/gru_cell_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_4/gru_cell_8/bias/v*
_output_shapes

:`*
dtype0
�
(Adam/gru_4/gru_cell_8/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*9
shared_name*(Adam/gru_4/gru_cell_8/recurrent_kernel/v
�
<Adam/gru_4/gru_cell_8/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/gru_4/gru_cell_8/recurrent_kernel/v*
_output_shapes

: `*
dtype0
�
Adam/gru_4/gru_cell_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@`*/
shared_name Adam/gru_4/gru_cell_8/kernel/v
�
2Adam/gru_4/gru_cell_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_4/gru_cell_8/kernel/v*
_output_shapes

:@`*
dtype0
�
+Adam/simple_rnn_5/simple_rnn_cell_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+Adam/simple_rnn_5/simple_rnn_cell_10/bias/v
�
?Adam/simple_rnn_5/simple_rnn_cell_10/bias/v/Read/ReadVariableOpReadVariableOp+Adam/simple_rnn_5/simple_rnn_cell_10/bias/v*
_output_shapes
:@*
dtype0
�
7Adam/simple_rnn_5/simple_rnn_cell_10/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*H
shared_name97Adam/simple_rnn_5/simple_rnn_cell_10/recurrent_kernel/v
�
KAdam/simple_rnn_5/simple_rnn_cell_10/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp7Adam/simple_rnn_5/simple_rnn_cell_10/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
�
-Adam/simple_rnn_5/simple_rnn_cell_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*>
shared_name/-Adam/simple_rnn_5/simple_rnn_cell_10/kernel/v
�
AAdam/simple_rnn_5/simple_rnn_cell_10/kernel/v/Read/ReadVariableOpReadVariableOp-Adam/simple_rnn_5/simple_rnn_cell_10/kernel/v*
_output_shapes

:@*
dtype0
�
Adam/dense_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_26/bias/v
y
(Adam/dense_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_26/kernel/v
�
*Adam/dense_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/v*
_output_shapes

: *
dtype0
�
Adam/gru_4/gru_cell_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*-
shared_nameAdam/gru_4/gru_cell_8/bias/m
�
0Adam/gru_4/gru_cell_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_4/gru_cell_8/bias/m*
_output_shapes

:`*
dtype0
�
(Adam/gru_4/gru_cell_8/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*9
shared_name*(Adam/gru_4/gru_cell_8/recurrent_kernel/m
�
<Adam/gru_4/gru_cell_8/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/gru_4/gru_cell_8/recurrent_kernel/m*
_output_shapes

: `*
dtype0
�
Adam/gru_4/gru_cell_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@`*/
shared_name Adam/gru_4/gru_cell_8/kernel/m
�
2Adam/gru_4/gru_cell_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_4/gru_cell_8/kernel/m*
_output_shapes

:@`*
dtype0
�
+Adam/simple_rnn_5/simple_rnn_cell_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+Adam/simple_rnn_5/simple_rnn_cell_10/bias/m
�
?Adam/simple_rnn_5/simple_rnn_cell_10/bias/m/Read/ReadVariableOpReadVariableOp+Adam/simple_rnn_5/simple_rnn_cell_10/bias/m*
_output_shapes
:@*
dtype0
�
7Adam/simple_rnn_5/simple_rnn_cell_10/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*H
shared_name97Adam/simple_rnn_5/simple_rnn_cell_10/recurrent_kernel/m
�
KAdam/simple_rnn_5/simple_rnn_cell_10/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp7Adam/simple_rnn_5/simple_rnn_cell_10/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
�
-Adam/simple_rnn_5/simple_rnn_cell_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*>
shared_name/-Adam/simple_rnn_5/simple_rnn_cell_10/kernel/m
�
AAdam/simple_rnn_5/simple_rnn_cell_10/kernel/m/Read/ReadVariableOpReadVariableOp-Adam/simple_rnn_5/simple_rnn_cell_10/kernel/m*
_output_shapes

:@*
dtype0
�
Adam/dense_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_26/bias/m
y
(Adam/dense_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_26/kernel/m
�
*Adam/dense_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/m*
_output_shapes

: *
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
�
gru_4/gru_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*&
shared_namegru_4/gru_cell_8/bias

)gru_4/gru_cell_8/bias/Read/ReadVariableOpReadVariableOpgru_4/gru_cell_8/bias*
_output_shapes

:`*
dtype0
�
!gru_4/gru_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*2
shared_name#!gru_4/gru_cell_8/recurrent_kernel
�
5gru_4/gru_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_4/gru_cell_8/recurrent_kernel*
_output_shapes

: `*
dtype0
�
gru_4/gru_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@`*(
shared_namegru_4/gru_cell_8/kernel
�
+gru_4/gru_cell_8/kernel/Read/ReadVariableOpReadVariableOpgru_4/gru_cell_8/kernel*
_output_shapes

:@`*
dtype0
�
$simple_rnn_5/simple_rnn_cell_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$simple_rnn_5/simple_rnn_cell_10/bias
�
8simple_rnn_5/simple_rnn_cell_10/bias/Read/ReadVariableOpReadVariableOp$simple_rnn_5/simple_rnn_cell_10/bias*
_output_shapes
:@*
dtype0
�
0simple_rnn_5/simple_rnn_cell_10/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*A
shared_name20simple_rnn_5/simple_rnn_cell_10/recurrent_kernel
�
Dsimple_rnn_5/simple_rnn_cell_10/recurrent_kernel/Read/ReadVariableOpReadVariableOp0simple_rnn_5/simple_rnn_cell_10/recurrent_kernel*
_output_shapes

:@@*
dtype0
�
&simple_rnn_5/simple_rnn_cell_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*7
shared_name(&simple_rnn_5/simple_rnn_cell_10/kernel
�
:simple_rnn_5/simple_rnn_cell_10/kernel/Read/ReadVariableOpReadVariableOp&simple_rnn_5/simple_rnn_cell_10/kernel*
_output_shapes

:@*
dtype0
r
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_26/bias
k
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes
:*
dtype0
z
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_26/kernel
s
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes

: *
dtype0
�
"serving_default_simple_rnn_5_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall"serving_default_simple_rnn_5_input&simple_rnn_5/simple_rnn_cell_10/kernel$simple_rnn_5/simple_rnn_cell_10/bias0simple_rnn_5/simple_rnn_cell_10/recurrent_kernelgru_4/gru_cell_8/kernel!gru_4/gru_cell_8/recurrent_kernelgru_4/gru_cell_8/biasdense_26/kerneldense_26/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_2211964

NoOpNoOp
�A
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�A
value�AB�A B�A
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
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
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
<
&0
'1
(2
)3
*4
+5
$6
%7*
<
&0
'1
(2
)3
*4
+5
$6
%7*
* 
�
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
1trace_0
2trace_1
3trace_2
4trace_3* 
6
5trace_0
6trace_1
7trace_2
8trace_3* 
* 
�
9iter

:beta_1

;beta_2
	<decay
=learning_rate$m�%m�&m�'m�(m�)m�*m�+m�$v�%v�&v�'v�(v�)v�*v�+v�*

>serving_default* 

&0
'1
(2*

&0
'1
(2*
* 
�

?states
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Etrace_0
Ftrace_1
Gtrace_2
Htrace_3* 
6
Itrace_0
Jtrace_1
Ktrace_2
Ltrace_3* 
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
S_random_generator

&kernel
'recurrent_kernel
(bias*
* 

)0
*1
+2*

)0
*1
+2*
* 
�

Tstates
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ztrace_0
[trace_1
\trace_2
]trace_3* 
6
^trace_0
_trace_1
`trace_2
atrace_3* 
* 
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
h_random_generator

)kernel
*recurrent_kernel
+bias*
* 

$0
%1*

$0
%1*
* 
�
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

ntrace_0* 

otrace_0* 
_Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_26/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&simple_rnn_5/simple_rnn_cell_10/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE0simple_rnn_5/simple_rnn_cell_10/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$simple_rnn_5/simple_rnn_cell_10/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgru_4/gru_cell_8/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!gru_4/gru_cell_8/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEgru_4/gru_cell_8/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

p0
q1*
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
&0
'1
(2*

&0
'1
(2*
* 
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

wtrace_0
xtrace_1* 

ytrace_0
ztrace_1* 
* 
* 
* 

0*
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
)0
*1
+2*

)0
*1
+2*
* 
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*
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
<
�	variables
�	keras_api

�total

�count*
<
�	variables
�	keras_api

�total

�count*
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

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_26/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_26/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/simple_rnn_5/simple_rnn_cell_10/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE7Adam/simple_rnn_5/simple_rnn_cell_10/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE+Adam/simple_rnn_5/simple_rnn_cell_10/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_4/gru_cell_8/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUE(Adam/gru_4/gru_cell_8/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/gru_4/gru_cell_8/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_26/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_26/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/simple_rnn_5/simple_rnn_cell_10/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE7Adam/simple_rnn_5/simple_rnn_cell_10/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE+Adam/simple_rnn_5/simple_rnn_cell_10/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_4/gru_cell_8/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUE(Adam/gru_4/gru_cell_8/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/gru_4/gru_cell_8/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOp:simple_rnn_5/simple_rnn_cell_10/kernel/Read/ReadVariableOpDsimple_rnn_5/simple_rnn_cell_10/recurrent_kernel/Read/ReadVariableOp8simple_rnn_5/simple_rnn_cell_10/bias/Read/ReadVariableOp+gru_4/gru_cell_8/kernel/Read/ReadVariableOp5gru_4/gru_cell_8/recurrent_kernel/Read/ReadVariableOp)gru_4/gru_cell_8/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_26/kernel/m/Read/ReadVariableOp(Adam/dense_26/bias/m/Read/ReadVariableOpAAdam/simple_rnn_5/simple_rnn_cell_10/kernel/m/Read/ReadVariableOpKAdam/simple_rnn_5/simple_rnn_cell_10/recurrent_kernel/m/Read/ReadVariableOp?Adam/simple_rnn_5/simple_rnn_cell_10/bias/m/Read/ReadVariableOp2Adam/gru_4/gru_cell_8/kernel/m/Read/ReadVariableOp<Adam/gru_4/gru_cell_8/recurrent_kernel/m/Read/ReadVariableOp0Adam/gru_4/gru_cell_8/bias/m/Read/ReadVariableOp*Adam/dense_26/kernel/v/Read/ReadVariableOp(Adam/dense_26/bias/v/Read/ReadVariableOpAAdam/simple_rnn_5/simple_rnn_cell_10/kernel/v/Read/ReadVariableOpKAdam/simple_rnn_5/simple_rnn_cell_10/recurrent_kernel/v/Read/ReadVariableOp?Adam/simple_rnn_5/simple_rnn_cell_10/bias/v/Read/ReadVariableOp2Adam/gru_4/gru_cell_8/kernel/v/Read/ReadVariableOp<Adam/gru_4/gru_cell_8/recurrent_kernel/v/Read/ReadVariableOp0Adam/gru_4/gru_cell_8/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
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
GPU 2J 8� *)
f$R"
 __inference__traced_save_2215217
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_26/kerneldense_26/bias&simple_rnn_5/simple_rnn_cell_10/kernel0simple_rnn_5/simple_rnn_cell_10/recurrent_kernel$simple_rnn_5/simple_rnn_cell_10/biasgru_4/gru_cell_8/kernel!gru_4/gru_cell_8/recurrent_kernelgru_4/gru_cell_8/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_26/kernel/mAdam/dense_26/bias/m-Adam/simple_rnn_5/simple_rnn_cell_10/kernel/m7Adam/simple_rnn_5/simple_rnn_cell_10/recurrent_kernel/m+Adam/simple_rnn_5/simple_rnn_cell_10/bias/mAdam/gru_4/gru_cell_8/kernel/m(Adam/gru_4/gru_cell_8/recurrent_kernel/mAdam/gru_4/gru_cell_8/bias/mAdam/dense_26/kernel/vAdam/dense_26/bias/v-Adam/simple_rnn_5/simple_rnn_cell_10/kernel/v7Adam/simple_rnn_5/simple_rnn_cell_10/recurrent_kernel/v+Adam/simple_rnn_5/simple_rnn_cell_10/bias/vAdam/gru_4/gru_cell_8/kernel/v(Adam/gru_4/gru_cell_8/recurrent_kernel/vAdam/gru_4/gru_cell_8/bias/v*-
Tin&
$2"*
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
GPU 2J 8� *,
f'R%
#__inference__traced_restore_2215326��1
�>
�
 __inference_standard_gru_2211004

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@B
ShapeShapetranspose:y:0*
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:���������`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:���������`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:��������� b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:��������� Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:��������� ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:��������� Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:��������� S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :��������� : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_2210914*
condR
while_cond_2210913*R
output_shapesA
?: : : : :��������� : : :@`:`: `:`*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:��������� ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:��������� X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_be92b9e3-7179-4c2b-9a03-8f4b43aafdda*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�	
�
while_cond_2213952
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2213952___redundant_placeholder05
1while_while_cond_2213952___redundant_placeholder15
1while_while_cond_2213952___redundant_placeholder25
1while_while_cond_2213952___redundant_placeholder35
1while_while_cond_2213952___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :��������� : :::::: 
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
:��������� :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
�>
�
 __inference_standard_gru_2214799

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@B
ShapeShapetranspose:y:0*
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:���������`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:���������`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:��������� b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:��������� Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:��������� ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:��������� Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:��������� S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :��������� : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_2214709*
condR
while_cond_2214708*R
output_shapesA
?: : : : :��������� : : :@`:`: `:`*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:��������� ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:��������� X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_e68a311f-c1d7-4943-85e8-55db03b37fe0*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�
�
.__inference_simple_rnn_5_layer_call_fn_2213015

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2210834s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_gru_4_layer_call_fn_2213480
inputs_0
unknown:@`
	unknown_0: `
	unknown_1:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_gru_4_layer_call_and_return_conditional_losses_2210711o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�
�
'__inference_gru_4_layer_call_fn_2213491

inputs
unknown:@`
	unknown_0: `
	unknown_1:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_gru_4_layer_call_and_return_conditional_losses_2211219o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�>
�
 __inference_standard_gru_2210496

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������@B
ShapeShapetranspose:y:0*
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:���������`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:���������`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:��������� b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:��������� Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:��������� ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:��������� Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:��������� S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :��������� : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_2210406*
condR
while_cond_2210405*R
output_shapesA
?: : : : :��������� : : :@`:`: `:`*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:��������� ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:��������� X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:������������������@:��������� :@`: `:`*<
api_implements*(gru_7d9efa0d-1eda-49ac-afce-88b95820ce05*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�4
�
)__inference_gpu_gru_with_fallback_2214119

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:�IU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*Q
_output_shapes?
=:������������������ :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:������������������@:��������� :@`: `:`*<
api_implements*(gru_b6541380-d387-4ad8-819e-2893f92843d6*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�4
�
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2209931

inputs,
simple_rnn_cell_10_2209856:@(
simple_rnn_cell_10_2209858:@,
simple_rnn_cell_10_2209860:@@
identity��*simple_rnn_cell_10/StatefulPartitionedCall�while;
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
valueB:�
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
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB:�
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
*simple_rnn_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_10_2209856simple_rnn_cell_10_2209858simple_rnn_cell_10_2209860*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_2209816n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_10_2209856simple_rnn_cell_10_2209858simple_rnn_cell_10_2209860*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2209868*
condR
while_cond_2209867*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������@{
NoOpNoOp+^simple_rnn_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2X
*simple_rnn_cell_10/StatefulPartitionedCall*simple_rnn_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
while_cond_2213391
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2213391___redundant_placeholder05
1while_while_cond_2213391___redundant_placeholder15
1while_while_cond_2213391___redundant_placeholder25
1while_while_cond_2213391___redundant_placeholder3
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
-: : : : :���������@: ::::: 
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
:���������@:

_output_shapes
: :

_output_shapes
:
�>
�
'__forward_gpu_gru_with_fallback_2212973

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:��������� :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_39d7b2a1-7cf4-4552-ab71-1e34684b15e2*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2212838_2212974*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�-
�
while_body_2214331
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:���������`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:���������`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:���������`�
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:���������`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:��������� Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:��������� t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:��������� ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:��������� o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:��������� k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:��������� U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:��������� l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:��������� P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:��������� c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:��������� h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:��������� r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:��������� "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :��������� : : :@`:`: `:`: 
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
:��������� :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
�>
�
 __inference_standard_gru_2210107

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������@B
ShapeShapetranspose:y:0*
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:���������`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:���������`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:��������� b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:��������� Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:��������� ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:��������� Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:��������� S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :��������� : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_2210017*
condR
while_cond_2210016*R
output_shapesA
?: : : : :��������� : : :@`:`: `:`*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:��������� ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:��������� X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:������������������@:��������� :@`: `:`*<
api_implements*(gru_e75d9f83-e331-4427-a235-3d066454014e*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�-
�
while_body_2213575
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:���������`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:���������`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:���������`�
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:���������`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:��������� Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:��������� t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:��������� ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:��������� o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:��������� k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:��������� U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:��������� l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:��������� P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:��������� c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:��������� h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:��������� r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:��������� "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :��������� : : :@`:`: `:`: 
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
:��������� :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
�
�
'__inference_gru_4_layer_call_fn_2213502

inputs
unknown:@`
	unknown_0: `
	unknown_1:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_gru_4_layer_call_and_return_conditional_losses_2211664o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
simple_rnn_5_while_cond_22125356
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_28
4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1O
Ksimple_rnn_5_while_simple_rnn_5_while_cond_2212535___redundant_placeholder0O
Ksimple_rnn_5_while_simple_rnn_5_while_cond_2212535___redundant_placeholder1O
Ksimple_rnn_5_while_simple_rnn_5_while_cond_2212535___redundant_placeholder2O
Ksimple_rnn_5_while_simple_rnn_5_while_cond_2212535___redundant_placeholder3
simple_rnn_5_while_identity
�
simple_rnn_5/while/LessLesssimple_rnn_5_while_placeholder4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: ::::: 
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
:���������@:

_output_shapes
: :

_output_shapes
:
�,
�
while_body_2213068
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:@F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@��/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp�.while/simple_rnn_cell_10/MatMul/ReadVariableOp�0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@y
while/simple_rnn_cell_10/TanhTanh while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_10/Tanh:y:0*
_output_shapes
: *
element_dtype0:���M
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
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_10/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp0^while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_10/MatMul/ReadVariableOp1^while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_10_matmul_readvariableop_resource9while_simple_rnn_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2b
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_10/MatMul/ReadVariableOp.while/simple_rnn_cell_10/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 
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
:���������@:

_output_shapes
: :

_output_shapes
: 
�-
�
while_body_2212183
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:���������`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:���������`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:���������`�
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:���������`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:��������� Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:��������� t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:��������� ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:��������� o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:��������� k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:��������� U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:��������� l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:��������� P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:��������� c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:��������� h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:��������� r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:��������� "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :��������� : : :@`:`: `:`: 
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
:��������� :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
�	
�
while_cond_2210016
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2210016___redundant_placeholder05
1while_while_cond_2210016___redundant_placeholder15
1while_while_cond_2210016___redundant_placeholder25
1while_while_cond_2210016___redundant_placeholder35
1while_while_cond_2210016___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :��������� : :::::: 
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
:��������� :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
�4
�
)__inference_gpu_gru_with_fallback_2211525

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:�IU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:��������� :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_f7f2608a-40e9-4efa-aef1-50ad684e039a*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�4
�
)__inference_gpu_gru_with_fallback_2212349

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:�IU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:��������� :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_4ee5b07d-b52c-4a92-9a69-4a47c2ebc525*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�>
�
'__forward_gpu_gru_with_fallback_2211216

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:��������� :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_be92b9e3-7179-4c2b-9a03-8f4b43aafdda*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2211081_2211217*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�-
�
while_body_2210017
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:���������`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:���������`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:���������`�
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:���������`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:��������� Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:��������� t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:��������� ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:��������� o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:��������� k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:��������� U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:��������� l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:��������� P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:��������� c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:��������� h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:��������� r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:��������� "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :��������� : : :@`:`: `:`: 
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
:��������� :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
��
�

<__inference___backward_gpu_gru_with_fallback_2212350_2212486
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:��������� d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:��������� `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:��������� O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:�
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:��������� q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:��������� �
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:��������� }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:��������� *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:���������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:�
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:��������� \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: �
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: �
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`�
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:���������@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:��������� e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:��������� :��������� :��������� : :��������� :��������� :��������� : ::���������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_4ee5b07d-b52c-4a92-9a69-4a47c2ebc525*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2212485*
go_backwards( *

time_major( :- )
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :-)
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:���������@:1
-
+
_output_shapes
:��������� :

_output_shapes
: :!

_output_shapes	
:�I: 

_output_shapes
::-)
'
_output_shapes
:��������� :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�	
�
while_cond_2211358
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2211358___redundant_placeholder05
1while_while_cond_2211358___redundant_placeholder15
1while_while_cond_2211358___redundant_placeholder25
1while_while_cond_2211358___redundant_placeholder35
1while_while_cond_2211358___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :��������� : :::::: 
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
:��������� :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
�	
�
while_cond_2213574
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2213574___redundant_placeholder05
1while_while_cond_2213574___redundant_placeholder15
1while_while_cond_2213574___redundant_placeholder25
1while_while_cond_2213574___redundant_placeholder35
1while_while_cond_2213574___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :��������� : :::::: 
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
:��������� :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
�4
�
)__inference_gpu_gru_with_fallback_2210572

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:�IU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*Q
_output_shapes?
=:������������������ :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:������������������@:��������� :@`: `:`*<
api_implements*(gru_7d9efa0d-1eda-49ac-afce-88b95820ce05*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�-
�
while_body_2210406
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:���������`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:���������`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:���������`�
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:���������`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:��������� Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:��������� t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:��������� ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:��������� o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:��������� k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:��������� U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:��������� l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:��������� P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:��������� c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:��������� h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:��������� r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:��������� "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :��������� : : :@`:`: `:`: 
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
:��������� :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
�=
�
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2213350

inputsC
1simple_rnn_cell_10_matmul_readvariableop_resource:@@
2simple_rnn_cell_10_biasadd_readvariableop_resource:@E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:@@
identity��)simple_rnn_cell_10/BiasAdd/ReadVariableOp�(simple_rnn_cell_10/MatMul/ReadVariableOp�*simple_rnn_cell_10/MatMul_1/ReadVariableOp�while;
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
valueB:�
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
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB:�
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@m
simple_rnn_cell_10/TanhTanhsimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2213284*
condR
while_cond_2213283*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
while_body_2210768
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:@F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@��/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp�.while/simple_rnn_cell_10/MatMul/ReadVariableOp�0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@y
while/simple_rnn_cell_10/TanhTanh while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_10/Tanh:y:0*
_output_shapes
: *
element_dtype0:���M
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
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_10/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp0^while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_10/MatMul/ReadVariableOp1^while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_10_matmul_readvariableop_resource9while_simple_rnn_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2b
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_10/MatMul/ReadVariableOp.while/simple_rnn_cell_10/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 
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
:���������@:

_output_shapes
: :

_output_shapes
: 
�>
�
'__forward_gpu_gru_with_fallback_2214255

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*Q
_output_shapes?
=:������������������ :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:������������������@:��������� :@`: `:`*<
api_implements*(gru_b6541380-d387-4ad8-819e-2893f92843d6*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2214120_2214256*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�>
�
'__forward_gpu_gru_with_fallback_2211661

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:��������� :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_f7f2608a-40e9-4efa-aef1-50ad684e039a*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2211526_2211662*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�>
�
'__forward_gpu_gru_with_fallback_2210708

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*Q
_output_shapes?
=:������������������ :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:������������������@:��������� :@`: `:`*<
api_implements*(gru_7d9efa0d-1eda-49ac-afce-88b95820ce05*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2210573_2210709*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�	
�
/__inference_sequential_18_layer_call_fn_2211889
simple_rnn_5_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@`
	unknown_3: `
	unknown_4:`
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_18_layer_call_and_return_conditional_losses_2211849o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:���������
,
_user_specified_namesimple_rnn_5_input
�
�
J__inference_sequential_18_layer_call_and_return_conditional_losses_2211849

inputs&
simple_rnn_5_2211829:@"
simple_rnn_5_2211831:@&
simple_rnn_5_2211833:@@
gru_4_2211836:@`
gru_4_2211838: `
gru_4_2211840:`"
dense_26_2211843: 
dense_26_2211845:
identity�� dense_26/StatefulPartitionedCall�gru_4/StatefulPartitionedCall�$simple_rnn_5/StatefulPartitionedCall�
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_5_2211829simple_rnn_5_2211831simple_rnn_5_2211833*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2211794�
gru_4/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0gru_4_2211836gru_4_2211838gru_4_2211840*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_gru_4_layer_call_and_return_conditional_losses_2211664�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall&gru_4/StatefulPartitionedCall:output:0dense_26_2211843dense_26_2211845*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2211237x
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_26/StatefulPartitionedCall^gru_4/StatefulPartitionedCall%^simple_rnn_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2>
gru_4/StatefulPartitionedCallgru_4/StatefulPartitionedCall2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
while_cond_2209336
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2209336___redundant_placeholder05
1while_while_cond_2209336___redundant_placeholder15
1while_while_cond_2209336___redundant_placeholder25
1while_while_cond_2209336___redundant_placeholder35
1while_while_cond_2209336___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :��������� : :::::: 
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
:��������� :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
�I
�
 __inference__traced_save_2215217
file_prefix.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableopE
Asavev2_simple_rnn_5_simple_rnn_cell_10_kernel_read_readvariableopO
Ksavev2_simple_rnn_5_simple_rnn_cell_10_recurrent_kernel_read_readvariableopC
?savev2_simple_rnn_5_simple_rnn_cell_10_bias_read_readvariableop6
2savev2_gru_4_gru_cell_8_kernel_read_readvariableop@
<savev2_gru_4_gru_cell_8_recurrent_kernel_read_readvariableop4
0savev2_gru_4_gru_cell_8_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_26_kernel_m_read_readvariableop3
/savev2_adam_dense_26_bias_m_read_readvariableopL
Hsavev2_adam_simple_rnn_5_simple_rnn_cell_10_kernel_m_read_readvariableopV
Rsavev2_adam_simple_rnn_5_simple_rnn_cell_10_recurrent_kernel_m_read_readvariableopJ
Fsavev2_adam_simple_rnn_5_simple_rnn_cell_10_bias_m_read_readvariableop=
9savev2_adam_gru_4_gru_cell_8_kernel_m_read_readvariableopG
Csavev2_adam_gru_4_gru_cell_8_recurrent_kernel_m_read_readvariableop;
7savev2_adam_gru_4_gru_cell_8_bias_m_read_readvariableop5
1savev2_adam_dense_26_kernel_v_read_readvariableop3
/savev2_adam_dense_26_bias_v_read_readvariableopL
Hsavev2_adam_simple_rnn_5_simple_rnn_cell_10_kernel_v_read_readvariableopV
Rsavev2_adam_simple_rnn_5_simple_rnn_cell_10_recurrent_kernel_v_read_readvariableopJ
Fsavev2_adam_simple_rnn_5_simple_rnn_cell_10_bias_v_read_readvariableop=
9savev2_adam_gru_4_gru_cell_8_kernel_v_read_readvariableopG
Csavev2_adam_gru_4_gru_cell_8_recurrent_kernel_v_read_readvariableop;
7savev2_adam_gru_4_gru_cell_8_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*�
value�B�"B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableopAsavev2_simple_rnn_5_simple_rnn_cell_10_kernel_read_readvariableopKsavev2_simple_rnn_5_simple_rnn_cell_10_recurrent_kernel_read_readvariableop?savev2_simple_rnn_5_simple_rnn_cell_10_bias_read_readvariableop2savev2_gru_4_gru_cell_8_kernel_read_readvariableop<savev2_gru_4_gru_cell_8_recurrent_kernel_read_readvariableop0savev2_gru_4_gru_cell_8_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_26_kernel_m_read_readvariableop/savev2_adam_dense_26_bias_m_read_readvariableopHsavev2_adam_simple_rnn_5_simple_rnn_cell_10_kernel_m_read_readvariableopRsavev2_adam_simple_rnn_5_simple_rnn_cell_10_recurrent_kernel_m_read_readvariableopFsavev2_adam_simple_rnn_5_simple_rnn_cell_10_bias_m_read_readvariableop9savev2_adam_gru_4_gru_cell_8_kernel_m_read_readvariableopCsavev2_adam_gru_4_gru_cell_8_recurrent_kernel_m_read_readvariableop7savev2_adam_gru_4_gru_cell_8_bias_m_read_readvariableop1savev2_adam_dense_26_kernel_v_read_readvariableop/savev2_adam_dense_26_bias_v_read_readvariableopHsavev2_adam_simple_rnn_5_simple_rnn_cell_10_kernel_v_read_readvariableopRsavev2_adam_simple_rnn_5_simple_rnn_cell_10_recurrent_kernel_v_read_readvariableopFsavev2_adam_simple_rnn_5_simple_rnn_cell_10_bias_v_read_readvariableop9savev2_adam_gru_4_gru_cell_8_kernel_v_read_readvariableopCsavev2_adam_gru_4_gru_cell_8_recurrent_kernel_v_read_readvariableop7savev2_adam_gru_4_gru_cell_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: : ::@:@@:@:@`: `:`: : : : : : : : : : ::@:@@:@:@`: `:`: ::@:@@:@:@`: `:`: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@`:$ 

_output_shapes

: `:$ 

_output_shapes

:`:	
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@`:$ 

_output_shapes

: `:$ 

_output_shapes

:`:$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@`:$  

_output_shapes

: `:$! 

_output_shapes

:`:"

_output_shapes
: 
Ɠ
�

<__inference___backward_gpu_gru_with_fallback_2214120_2214256
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:��������� d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:��������� `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:��������� O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:�
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:��������� q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:��������� �
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:��������� }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :������������������ *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*T
_output_shapesB
@:������������������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :������������������@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:�
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:��������� \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: �
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: �
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`�
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :������������������@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:��������� e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:��������� :��������� :��������� : :��������� :��������� :������������������ : ::������������������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_b6541380-d387-4ad8-819e-2893f92843d6*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2214255*
go_backwards( *

time_major( :- )
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :-)
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� ::6
4
_output_shapes"
 :������������������ :

_output_shapes
: :

_output_shapes
:::	6
4
_output_shapes"
 :������������������@:1
-
+
_output_shapes
:��������� :

_output_shapes
: :!

_output_shapes	
:�I: 

_output_shapes
::-)
'
_output_shapes
:��������� :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�	
�
E__inference_dense_26_layer_call_and_return_conditional_losses_2211237

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�>
�
 __inference_standard_gru_2212273

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@B
ShapeShapetranspose:y:0*
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:���������`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:���������`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:��������� b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:��������� Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:��������� ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:��������� Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:��������� S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :��������� : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_2212183*
condR
while_cond_2212182*R
output_shapesA
?: : : : :��������� : : :@`:`: `:`*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:��������� ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:��������� X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_4ee5b07d-b52c-4a92-9a69-4a47c2ebc525*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�-
�
while_body_2210914
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:���������`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:���������`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:���������`�
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:���������`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:��������� Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:��������� t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:��������� ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:��������� o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:��������� k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:��������� U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:��������� l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:��������� P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:��������� c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:��������� h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:��������� r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:��������� "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :��������� : : :@`:`: `:`: 
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
:��������� :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
��
�

<__inference___backward_gpu_gru_with_fallback_2211526_2211662
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:��������� d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:��������� `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:��������� O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:�
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:��������� q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:��������� �
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:��������� }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:��������� *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:���������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:�
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:��������� \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: �
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: �
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`�
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:���������@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:��������� e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:��������� :��������� :��������� : :��������� :��������� :��������� : ::���������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_f7f2608a-40e9-4efa-aef1-50ad684e039a*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2211661*
go_backwards( *

time_major( :- )
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :-)
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:���������@:1
-
+
_output_shapes
:��������� :

_output_shapes
: :!

_output_shapes	
:�I: 

_output_shapes
::-)
'
_output_shapes
:��������� :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
*__inference_dense_26_layer_call_fn_2215023

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2211237o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�=
�
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2213458

inputsC
1simple_rnn_cell_10_matmul_readvariableop_resource:@@
2simple_rnn_cell_10_biasadd_readvariableop_resource:@E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:@@
identity��)simple_rnn_cell_10/BiasAdd/ReadVariableOp�(simple_rnn_cell_10/MatMul/ReadVariableOp�*simple_rnn_cell_10/MatMul_1/ReadVariableOp�while;
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
valueB:�
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
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB:�
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@m
simple_rnn_cell_10/TanhTanhsimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2213392*
condR
while_cond_2213391*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�-
�
while_body_2211359
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:���������`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:���������`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:���������`�
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:���������`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:��������� Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:��������� t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:��������� ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:��������� o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:��������� k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:��������� U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:��������� l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:��������� P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:��������� c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:��������� h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:��������� r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:��������� "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :��������� : : :@`:`: `:`: 
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
:��������� :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
�4
�
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2209772

inputs,
simple_rnn_cell_10_2209697:@(
simple_rnn_cell_10_2209699:@,
simple_rnn_cell_10_2209701:@@
identity��*simple_rnn_cell_10/StatefulPartitionedCall�while;
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
valueB:�
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
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB:�
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
*simple_rnn_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_10_2209697simple_rnn_cell_10_2209699simple_rnn_cell_10_2209701*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_2209696n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_10_2209697simple_rnn_cell_10_2209699simple_rnn_cell_10_2209701*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2209709*
condR
while_cond_2209708*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������@{
NoOpNoOp+^simple_rnn_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2X
*simple_rnn_cell_10/StatefulPartitionedCall*simple_rnn_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�|
�	
"__inference__wrapped_model_2209648
simple_rnn_5_input^
Lsequential_18_simple_rnn_5_simple_rnn_cell_10_matmul_readvariableop_resource:@[
Msequential_18_simple_rnn_5_simple_rnn_cell_10_biasadd_readvariableop_resource:@`
Nsequential_18_simple_rnn_5_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@B
0sequential_18_gru_4_read_readvariableop_resource:@`D
2sequential_18_gru_4_read_1_readvariableop_resource: `D
2sequential_18_gru_4_read_2_readvariableop_resource:`G
5sequential_18_dense_26_matmul_readvariableop_resource: D
6sequential_18_dense_26_biasadd_readvariableop_resource:
identity��-sequential_18/dense_26/BiasAdd/ReadVariableOp�,sequential_18/dense_26/MatMul/ReadVariableOp�'sequential_18/gru_4/Read/ReadVariableOp�)sequential_18/gru_4/Read_1/ReadVariableOp�)sequential_18/gru_4/Read_2/ReadVariableOp�Dsequential_18/simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOp�Csequential_18/simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOp�Esequential_18/simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOp� sequential_18/simple_rnn_5/whileb
 sequential_18/simple_rnn_5/ShapeShapesimple_rnn_5_input*
T0*
_output_shapes
:x
.sequential_18/simple_rnn_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential_18/simple_rnn_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential_18/simple_rnn_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(sequential_18/simple_rnn_5/strided_sliceStridedSlice)sequential_18/simple_rnn_5/Shape:output:07sequential_18/simple_rnn_5/strided_slice/stack:output:09sequential_18/simple_rnn_5/strided_slice/stack_1:output:09sequential_18/simple_rnn_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_18/simple_rnn_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@�
'sequential_18/simple_rnn_5/zeros/packedPack1sequential_18/simple_rnn_5/strided_slice:output:02sequential_18/simple_rnn_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&sequential_18/simple_rnn_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
 sequential_18/simple_rnn_5/zerosFill0sequential_18/simple_rnn_5/zeros/packed:output:0/sequential_18/simple_rnn_5/zeros/Const:output:0*
T0*'
_output_shapes
:���������@~
)sequential_18/simple_rnn_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
$sequential_18/simple_rnn_5/transpose	Transposesimple_rnn_5_input2sequential_18/simple_rnn_5/transpose/perm:output:0*
T0*+
_output_shapes
:���������z
"sequential_18/simple_rnn_5/Shape_1Shape(sequential_18/simple_rnn_5/transpose:y:0*
T0*
_output_shapes
:z
0sequential_18/simple_rnn_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_18/simple_rnn_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_18/simple_rnn_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*sequential_18/simple_rnn_5/strided_slice_1StridedSlice+sequential_18/simple_rnn_5/Shape_1:output:09sequential_18/simple_rnn_5/strided_slice_1/stack:output:0;sequential_18/simple_rnn_5/strided_slice_1/stack_1:output:0;sequential_18/simple_rnn_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
6sequential_18/simple_rnn_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
(sequential_18/simple_rnn_5/TensorArrayV2TensorListReserve?sequential_18/simple_rnn_5/TensorArrayV2/element_shape:output:03sequential_18/simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Psequential_18/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Bsequential_18/simple_rnn_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_18/simple_rnn_5/transpose:y:0Ysequential_18/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���z
0sequential_18/simple_rnn_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_18/simple_rnn_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_18/simple_rnn_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*sequential_18/simple_rnn_5/strided_slice_2StridedSlice(sequential_18/simple_rnn_5/transpose:y:09sequential_18/simple_rnn_5/strided_slice_2/stack:output:0;sequential_18/simple_rnn_5/strided_slice_2/stack_1:output:0;sequential_18/simple_rnn_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
Csequential_18/simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpLsequential_18_simple_rnn_5_simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
4sequential_18/simple_rnn_5/simple_rnn_cell_10/MatMulMatMul3sequential_18/simple_rnn_5/strided_slice_2:output:0Ksequential_18/simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Dsequential_18/simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpMsequential_18_simple_rnn_5_simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
5sequential_18/simple_rnn_5/simple_rnn_cell_10/BiasAddBiasAdd>sequential_18/simple_rnn_5/simple_rnn_cell_10/MatMul:product:0Lsequential_18/simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Esequential_18/simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpNsequential_18_simple_rnn_5_simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
6sequential_18/simple_rnn_5/simple_rnn_cell_10/MatMul_1MatMul)sequential_18/simple_rnn_5/zeros:output:0Msequential_18/simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
1sequential_18/simple_rnn_5/simple_rnn_cell_10/addAddV2>sequential_18/simple_rnn_5/simple_rnn_cell_10/BiasAdd:output:0@sequential_18/simple_rnn_5/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
2sequential_18/simple_rnn_5/simple_rnn_cell_10/TanhTanh5sequential_18/simple_rnn_5/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@�
8sequential_18/simple_rnn_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
*sequential_18/simple_rnn_5/TensorArrayV2_1TensorListReserveAsequential_18/simple_rnn_5/TensorArrayV2_1/element_shape:output:03sequential_18/simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���a
sequential_18/simple_rnn_5/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3sequential_18/simple_rnn_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������o
-sequential_18/simple_rnn_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
 sequential_18/simple_rnn_5/whileWhile6sequential_18/simple_rnn_5/while/loop_counter:output:0<sequential_18/simple_rnn_5/while/maximum_iterations:output:0(sequential_18/simple_rnn_5/time:output:03sequential_18/simple_rnn_5/TensorArrayV2_1:handle:0)sequential_18/simple_rnn_5/zeros:output:03sequential_18/simple_rnn_5/strided_slice_1:output:0Rsequential_18/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0Lsequential_18_simple_rnn_5_simple_rnn_cell_10_matmul_readvariableop_resourceMsequential_18_simple_rnn_5_simple_rnn_cell_10_biasadd_readvariableop_resourceNsequential_18_simple_rnn_5_simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *9
body1R/
-sequential_18_simple_rnn_5_while_body_2209202*9
cond1R/
-sequential_18_simple_rnn_5_while_cond_2209201*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
Ksequential_18/simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
=sequential_18/simple_rnn_5/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_18/simple_rnn_5/while:output:3Tsequential_18/simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0�
0sequential_18/simple_rnn_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������|
2sequential_18/simple_rnn_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2sequential_18/simple_rnn_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*sequential_18/simple_rnn_5/strided_slice_3StridedSliceFsequential_18/simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:09sequential_18/simple_rnn_5/strided_slice_3/stack:output:0;sequential_18/simple_rnn_5/strided_slice_3/stack_1:output:0;sequential_18/simple_rnn_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask�
+sequential_18/simple_rnn_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
&sequential_18/simple_rnn_5/transpose_1	TransposeFsequential_18/simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:04sequential_18/simple_rnn_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@s
sequential_18/gru_4/ShapeShape*sequential_18/simple_rnn_5/transpose_1:y:0*
T0*
_output_shapes
:q
'sequential_18/gru_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_18/gru_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_18/gru_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!sequential_18/gru_4/strided_sliceStridedSlice"sequential_18/gru_4/Shape:output:00sequential_18/gru_4/strided_slice/stack:output:02sequential_18/gru_4/strided_slice/stack_1:output:02sequential_18/gru_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_18/gru_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
 sequential_18/gru_4/zeros/packedPack*sequential_18/gru_4/strided_slice:output:0+sequential_18/gru_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_18/gru_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_18/gru_4/zerosFill)sequential_18/gru_4/zeros/packed:output:0(sequential_18/gru_4/zeros/Const:output:0*
T0*'
_output_shapes
:��������� �
'sequential_18/gru_4/Read/ReadVariableOpReadVariableOp0sequential_18_gru_4_read_readvariableop_resource*
_output_shapes

:@`*
dtype0�
sequential_18/gru_4/IdentityIdentity/sequential_18/gru_4/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:@`�
)sequential_18/gru_4/Read_1/ReadVariableOpReadVariableOp2sequential_18_gru_4_read_1_readvariableop_resource*
_output_shapes

: `*
dtype0�
sequential_18/gru_4/Identity_1Identity1sequential_18/gru_4/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `�
)sequential_18/gru_4/Read_2/ReadVariableOpReadVariableOp2sequential_18_gru_4_read_2_readvariableop_resource*
_output_shapes

:`*
dtype0�
sequential_18/gru_4/Identity_2Identity1sequential_18/gru_4/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`�
#sequential_18/gru_4/PartitionedCallPartitionedCall*sequential_18/simple_rnn_5/transpose_1:y:0"sequential_18/gru_4/zeros:output:0%sequential_18/gru_4/Identity:output:0'sequential_18/gru_4/Identity_1:output:0'sequential_18/gru_4/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:��������� :��������� :��������� : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2209427�
,sequential_18/dense_26/MatMul/ReadVariableOpReadVariableOp5sequential_18_dense_26_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
sequential_18/dense_26/MatMulMatMul,sequential_18/gru_4/PartitionedCall:output:04sequential_18/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-sequential_18/dense_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_18/dense_26/BiasAddBiasAdd'sequential_18/dense_26/MatMul:product:05sequential_18/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
IdentityIdentity'sequential_18/dense_26/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^sequential_18/dense_26/BiasAdd/ReadVariableOp-^sequential_18/dense_26/MatMul/ReadVariableOp(^sequential_18/gru_4/Read/ReadVariableOp*^sequential_18/gru_4/Read_1/ReadVariableOp*^sequential_18/gru_4/Read_2/ReadVariableOpE^sequential_18/simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOpD^sequential_18/simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOpF^sequential_18/simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOp!^sequential_18/simple_rnn_5/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2^
-sequential_18/dense_26/BiasAdd/ReadVariableOp-sequential_18/dense_26/BiasAdd/ReadVariableOp2\
,sequential_18/dense_26/MatMul/ReadVariableOp,sequential_18/dense_26/MatMul/ReadVariableOp2R
'sequential_18/gru_4/Read/ReadVariableOp'sequential_18/gru_4/Read/ReadVariableOp2V
)sequential_18/gru_4/Read_1/ReadVariableOp)sequential_18/gru_4/Read_1/ReadVariableOp2V
)sequential_18/gru_4/Read_2/ReadVariableOp)sequential_18/gru_4/Read_2/ReadVariableOp2�
Dsequential_18/simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOpDsequential_18/simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOp2�
Csequential_18/simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOpCsequential_18/simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOp2�
Esequential_18/simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOpEsequential_18/simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOp2D
 sequential_18/simple_rnn_5/while sequential_18/simple_rnn_5/while:_ [
+
_output_shapes
:���������
,
_user_specified_namesimple_rnn_5_input
�!
�
while_body_2209868
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_10_2209890_0:@0
"while_simple_rnn_cell_10_2209892_0:@4
"while_simple_rnn_cell_10_2209894_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_10_2209890:@.
 while_simple_rnn_cell_10_2209892:@2
 while_simple_rnn_cell_10_2209894:@@��0while/simple_rnn_cell_10/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
0while/simple_rnn_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_10_2209890_0"while_simple_rnn_cell_10_2209892_0"while_simple_rnn_cell_10_2209894_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_2209816�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_10/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
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
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity9while/simple_rnn_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@

while/NoOpNoOp1^while/simple_rnn_cell_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_10_2209890"while_simple_rnn_cell_10_2209890_0"F
 while_simple_rnn_cell_10_2209892"while_simple_rnn_cell_10_2209892_0"F
 while_simple_rnn_cell_10_2209894"while_simple_rnn_cell_10_2209894_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2d
0while/simple_rnn_cell_10/StatefulPartitionedCall0while/simple_rnn_cell_10/StatefulPartitionedCall: 
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
:���������@:

_output_shapes
: :

_output_shapes
: 
�,
�
while_body_2213284
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:@F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@��/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp�.while/simple_rnn_cell_10/MatMul/ReadVariableOp�0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@y
while/simple_rnn_cell_10/TanhTanh while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_10/Tanh:y:0*
_output_shapes
: *
element_dtype0:���M
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
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_10/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp0^while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_10/MatMul/ReadVariableOp1^while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_10_matmul_readvariableop_resource9while_simple_rnn_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2b
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_10/MatMul/ReadVariableOp.while/simple_rnn_cell_10/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 
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
:���������@:

_output_shapes
: :

_output_shapes
: 
�8
�
simple_rnn_5_while_body_22125366
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_25
1simple_rnn_5_while_simple_rnn_5_strided_slice_1_0q
msimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0X
Fsimple_rnn_5_while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@U
Gsimple_rnn_5_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@Z
Hsimple_rnn_5_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@
simple_rnn_5_while_identity!
simple_rnn_5_while_identity_1!
simple_rnn_5_while_identity_2!
simple_rnn_5_while_identity_3!
simple_rnn_5_while_identity_43
/simple_rnn_5_while_simple_rnn_5_strided_slice_1o
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensorV
Dsimple_rnn_5_while_simple_rnn_cell_10_matmul_readvariableop_resource:@S
Esimple_rnn_5_while_simple_rnn_cell_10_biasadd_readvariableop_resource:@X
Fsimple_rnn_5_while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@��<simple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp�;simple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOp�=simple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp�
Dsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
6simple_rnn_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_5_while_placeholderMsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
;simple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpFsimple_rnn_5_while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
,simple_rnn_5/while/simple_rnn_cell_10/MatMulMatMul=simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem:item:0Csimple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<simple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpGsimple_rnn_5_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
-simple_rnn_5/while/simple_rnn_cell_10/BiasAddBiasAdd6simple_rnn_5/while/simple_rnn_cell_10/MatMul:product:0Dsimple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
=simple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpHsimple_rnn_5_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
.simple_rnn_5/while/simple_rnn_cell_10/MatMul_1MatMul simple_rnn_5_while_placeholder_2Esimple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)simple_rnn_5/while/simple_rnn_cell_10/addAddV26simple_rnn_5/while/simple_rnn_cell_10/BiasAdd:output:08simple_rnn_5/while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
*simple_rnn_5/while/simple_rnn_cell_10/TanhTanh-simple_rnn_5/while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@�
7simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_5_while_placeholder_1simple_rnn_5_while_placeholder.simple_rnn_5/while/simple_rnn_cell_10/Tanh:y:0*
_output_shapes
: *
element_dtype0:���Z
simple_rnn_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_5/while/addAddV2simple_rnn_5_while_placeholder!simple_rnn_5/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_5/while/add_1AddV22simple_rnn_5_while_simple_rnn_5_while_loop_counter#simple_rnn_5/while/add_1/y:output:0*
T0*
_output_shapes
: �
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/add_1:z:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_5/while/Identity_1Identity8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_5/while/Identity_2Identitysimple_rnn_5/while/add:z:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_5/while/Identity_3IdentityGsimple_rnn_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_5/while/Identity_4Identity.simple_rnn_5/while/simple_rnn_cell_10/Tanh:y:0^simple_rnn_5/while/NoOp*
T0*'
_output_shapes
:���������@�
simple_rnn_5/while/NoOpNoOp=^simple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp<^simple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOp>^simple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0"G
simple_rnn_5_while_identity_1&simple_rnn_5/while/Identity_1:output:0"G
simple_rnn_5_while_identity_2&simple_rnn_5/while/Identity_2:output:0"G
simple_rnn_5_while_identity_3&simple_rnn_5/while/Identity_3:output:0"G
simple_rnn_5_while_identity_4&simple_rnn_5/while/Identity_4:output:0"d
/simple_rnn_5_while_simple_rnn_5_strided_slice_11simple_rnn_5_while_simple_rnn_5_strided_slice_1_0"�
Esimple_rnn_5_while_simple_rnn_cell_10_biasadd_readvariableop_resourceGsimple_rnn_5_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"�
Fsimple_rnn_5_while_simple_rnn_cell_10_matmul_1_readvariableop_resourceHsimple_rnn_5_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"�
Dsimple_rnn_5_while_simple_rnn_cell_10_matmul_readvariableop_resourceFsimple_rnn_5_while_simple_rnn_cell_10_matmul_readvariableop_resource_0"�
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensormsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2|
<simple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp<simple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2z
;simple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOp;simple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOp2~
=simple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp=simple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 
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
:���������@:

_output_shapes
: :

_output_shapes
: 
Ɠ
�

<__inference___backward_gpu_gru_with_fallback_2213742_2213878
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:��������� d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:��������� `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:��������� O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:�
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:��������� q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:��������� �
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:��������� }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :������������������ *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*T
_output_shapesB
@:������������������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :������������������@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:�
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:��������� \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: �
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: �
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`�
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :������������������@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:��������� e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:��������� :��������� :��������� : :��������� :��������� :������������������ : ::������������������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_ceb15964-ea3b-4bb7-83fb-6d4227c7beed*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2213877*
go_backwards( *

time_major( :- )
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :-)
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� ::6
4
_output_shapes"
 :������������������ :

_output_shapes
: :

_output_shapes
:::	6
4
_output_shapes"
 :������������������@:1
-
+
_output_shapes
:��������� :

_output_shapes
: :!

_output_shapes	
:�I: 

_output_shapes
::-)
'
_output_shapes
:��������� :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�>
�
'__forward_gpu_gru_with_fallback_2215011

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:��������� :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_e68a311f-c1d7-4943-85e8-55db03b37fe0*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2214876_2215012*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�>
�
'__forward_gpu_gru_with_fallback_2212485

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:��������� :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_4ee5b07d-b52c-4a92-9a69-4a47c2ebc525*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2212350_2212486*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�
�
.__inference_simple_rnn_5_layer_call_fn_2213004
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2209931|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_2209816

inputs

states0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:���������@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_namestates
�
�
J__inference_sequential_18_layer_call_and_return_conditional_losses_2211935
simple_rnn_5_input&
simple_rnn_5_2211915:@"
simple_rnn_5_2211917:@&
simple_rnn_5_2211919:@@
gru_4_2211922:@`
gru_4_2211924: `
gru_4_2211926:`"
dense_26_2211929: 
dense_26_2211931:
identity�� dense_26/StatefulPartitionedCall�gru_4/StatefulPartitionedCall�$simple_rnn_5/StatefulPartitionedCall�
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_5_inputsimple_rnn_5_2211915simple_rnn_5_2211917simple_rnn_5_2211919*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2211794�
gru_4/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0gru_4_2211922gru_4_2211924gru_4_2211926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_gru_4_layer_call_and_return_conditional_losses_2211664�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall&gru_4/StatefulPartitionedCall:output:0dense_26_2211929dense_26_2211931*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2211237x
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_26/StatefulPartitionedCall^gru_4/StatefulPartitionedCall%^simple_rnn_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2>
gru_4/StatefulPartitionedCallgru_4/StatefulPartitionedCall2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall:_ [
+
_output_shapes
:���������
,
_user_specified_namesimple_rnn_5_input
�	
�
while_cond_2214708
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2214708___redundant_placeholder05
1while_while_cond_2214708___redundant_placeholder15
1while_while_cond_2214708___redundant_placeholder25
1while_while_cond_2214708___redundant_placeholder35
1while_while_cond_2214708___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :��������� : :::::: 
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
:��������� :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
�
�
-sequential_18_simple_rnn_5_while_cond_2209201R
Nsequential_18_simple_rnn_5_while_sequential_18_simple_rnn_5_while_loop_counterX
Tsequential_18_simple_rnn_5_while_sequential_18_simple_rnn_5_while_maximum_iterations0
,sequential_18_simple_rnn_5_while_placeholder2
.sequential_18_simple_rnn_5_while_placeholder_12
.sequential_18_simple_rnn_5_while_placeholder_2T
Psequential_18_simple_rnn_5_while_less_sequential_18_simple_rnn_5_strided_slice_1k
gsequential_18_simple_rnn_5_while_sequential_18_simple_rnn_5_while_cond_2209201___redundant_placeholder0k
gsequential_18_simple_rnn_5_while_sequential_18_simple_rnn_5_while_cond_2209201___redundant_placeholder1k
gsequential_18_simple_rnn_5_while_sequential_18_simple_rnn_5_while_cond_2209201___redundant_placeholder2k
gsequential_18_simple_rnn_5_while_sequential_18_simple_rnn_5_while_cond_2209201___redundant_placeholder3-
)sequential_18_simple_rnn_5_while_identity
�
%sequential_18/simple_rnn_5/while/LessLess,sequential_18_simple_rnn_5_while_placeholderPsequential_18_simple_rnn_5_while_less_sequential_18_simple_rnn_5_strided_slice_1*
T0*
_output_shapes
: �
)sequential_18/simple_rnn_5/while/IdentityIdentity)sequential_18/simple_rnn_5/while/Less:z:0*
T0
*
_output_shapes
: "_
)sequential_18_simple_rnn_5_while_identity2sequential_18/simple_rnn_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: ::::: 
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
:���������@:

_output_shapes
: :

_output_shapes
:
�-
�
while_body_2212671
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:���������`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:���������`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:���������`�
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:���������`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:��������� Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:��������� t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:��������� ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:��������� o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:��������� k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:��������� U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:��������� l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:��������� P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:��������� c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:��������� h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:��������� r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:��������� "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :��������� : : :@`:`: `:`: 
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
:��������� :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
Ɠ
�

<__inference___backward_gpu_gru_with_fallback_2210573_2210709
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:��������� d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:��������� `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:��������� O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:�
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:��������� q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:��������� �
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:��������� }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :������������������ *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*T
_output_shapesB
@:������������������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :������������������@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:�
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:��������� \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: �
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: �
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`�
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :������������������@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:��������� e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:��������� :��������� :��������� : :��������� :��������� :������������������ : ::������������������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_7d9efa0d-1eda-49ac-afce-88b95820ce05*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2210708*
go_backwards( *

time_major( :- )
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :-)
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� ::6
4
_output_shapes"
 :������������������ :

_output_shapes
: :

_output_shapes
:::	6
4
_output_shapes"
 :������������������@:1
-
+
_output_shapes
:��������� :

_output_shapes
: :!

_output_shapes	
:�I: 

_output_shapes
::-)
'
_output_shapes
:��������� :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�8
�
simple_rnn_5_while_body_22120486
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_25
1simple_rnn_5_while_simple_rnn_5_strided_slice_1_0q
msimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0X
Fsimple_rnn_5_while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@U
Gsimple_rnn_5_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@Z
Hsimple_rnn_5_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@
simple_rnn_5_while_identity!
simple_rnn_5_while_identity_1!
simple_rnn_5_while_identity_2!
simple_rnn_5_while_identity_3!
simple_rnn_5_while_identity_43
/simple_rnn_5_while_simple_rnn_5_strided_slice_1o
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensorV
Dsimple_rnn_5_while_simple_rnn_cell_10_matmul_readvariableop_resource:@S
Esimple_rnn_5_while_simple_rnn_cell_10_biasadd_readvariableop_resource:@X
Fsimple_rnn_5_while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@��<simple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp�;simple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOp�=simple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp�
Dsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
6simple_rnn_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_5_while_placeholderMsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
;simple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpFsimple_rnn_5_while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
,simple_rnn_5/while/simple_rnn_cell_10/MatMulMatMul=simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem:item:0Csimple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<simple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpGsimple_rnn_5_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
-simple_rnn_5/while/simple_rnn_cell_10/BiasAddBiasAdd6simple_rnn_5/while/simple_rnn_cell_10/MatMul:product:0Dsimple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
=simple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpHsimple_rnn_5_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
.simple_rnn_5/while/simple_rnn_cell_10/MatMul_1MatMul simple_rnn_5_while_placeholder_2Esimple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)simple_rnn_5/while/simple_rnn_cell_10/addAddV26simple_rnn_5/while/simple_rnn_cell_10/BiasAdd:output:08simple_rnn_5/while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
*simple_rnn_5/while/simple_rnn_cell_10/TanhTanh-simple_rnn_5/while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@�
7simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_5_while_placeholder_1simple_rnn_5_while_placeholder.simple_rnn_5/while/simple_rnn_cell_10/Tanh:y:0*
_output_shapes
: *
element_dtype0:���Z
simple_rnn_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_5/while/addAddV2simple_rnn_5_while_placeholder!simple_rnn_5/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_5/while/add_1AddV22simple_rnn_5_while_simple_rnn_5_while_loop_counter#simple_rnn_5/while/add_1/y:output:0*
T0*
_output_shapes
: �
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/add_1:z:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_5/while/Identity_1Identity8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_5/while/Identity_2Identitysimple_rnn_5/while/add:z:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_5/while/Identity_3IdentityGsimple_rnn_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_5/while/Identity_4Identity.simple_rnn_5/while/simple_rnn_cell_10/Tanh:y:0^simple_rnn_5/while/NoOp*
T0*'
_output_shapes
:���������@�
simple_rnn_5/while/NoOpNoOp=^simple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp<^simple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOp>^simple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0"G
simple_rnn_5_while_identity_1&simple_rnn_5/while/Identity_1:output:0"G
simple_rnn_5_while_identity_2&simple_rnn_5/while/Identity_2:output:0"G
simple_rnn_5_while_identity_3&simple_rnn_5/while/Identity_3:output:0"G
simple_rnn_5_while_identity_4&simple_rnn_5/while/Identity_4:output:0"d
/simple_rnn_5_while_simple_rnn_5_strided_slice_11simple_rnn_5_while_simple_rnn_5_strided_slice_1_0"�
Esimple_rnn_5_while_simple_rnn_cell_10_biasadd_readvariableop_resourceGsimple_rnn_5_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"�
Fsimple_rnn_5_while_simple_rnn_cell_10_matmul_1_readvariableop_resourceHsimple_rnn_5_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"�
Dsimple_rnn_5_while_simple_rnn_cell_10_matmul_readvariableop_resourceFsimple_rnn_5_while_simple_rnn_cell_10_matmul_readvariableop_resource_0"�
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensormsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2|
<simple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp<simple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2z
;simple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOp;simple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOp2~
=simple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp=simple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 
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
:���������@:

_output_shapes
: :

_output_shapes
: 
�	
�
E__inference_dense_26_layer_call_and_return_conditional_losses_2215033

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�>
�
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2213134
inputs_0C
1simple_rnn_cell_10_matmul_readvariableop_resource:@@
2simple_rnn_cell_10_biasadd_readvariableop_resource:@E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:@@
identity��)simple_rnn_cell_10/BiasAdd/ReadVariableOp�(simple_rnn_cell_10/MatMul/ReadVariableOp�*simple_rnn_cell_10/MatMul_1/ReadVariableOp�while=
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
valueB:�
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
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB:�
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@m
simple_rnn_cell_10/TanhTanhsimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2213068*
condR
while_cond_2213067*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������@�
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�	
�
/__inference_sequential_18_layer_call_fn_2211263
simple_rnn_5_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@`
	unknown_3: `
	unknown_4:`
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_18_layer_call_and_return_conditional_losses_2211244o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:���������
,
_user_specified_namesimple_rnn_5_input
�
�
while_cond_2209867
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2209867___redundant_placeholder05
1while_while_cond_2209867___redundant_placeholder15
1while_while_cond_2209867___redundant_placeholder25
1while_while_cond_2209867___redundant_placeholder3
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
-: : : : :���������@: ::::: 
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
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
B__inference_gru_4_layer_call_and_return_conditional_losses_2214636

inputs.
read_readvariableop_resource:@`0
read_1_readvariableop_resource: `0
read_2_readvariableop_resource:`

identity_3��Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp;
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
valueB:�
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
value	B : s
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
:��������� p
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:@`*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:@`t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `t
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype0^

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`�
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:��������� :��������� :��������� : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2214421i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
B__inference_gru_4_layer_call_and_return_conditional_losses_2210322

inputs.
read_readvariableop_resource:@`0
read_1_readvariableop_resource: `0
read_2_readvariableop_resource:`

identity_3��Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp;
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
valueB:�
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
value	B : s
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
:��������� p
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:@`*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:@`t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `t
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype0^

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`�
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:��������� :��������� :��������� : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2210107i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�	
�
/__inference_sequential_18_layer_call_fn_2211985

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@`
	unknown_3: `
	unknown_4:`
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_18_layer_call_and_return_conditional_losses_2211244o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_2213283
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2213283___redundant_placeholder05
1while_while_cond_2213283___redundant_placeholder15
1while_while_cond_2213283___redundant_placeholder25
1while_while_cond_2213283___redundant_placeholder3
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
-: : : : :���������@: ::::: 
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
:���������@:

_output_shapes
: :

_output_shapes
:
�!
�
while_body_2209709
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_10_2209731_0:@0
"while_simple_rnn_cell_10_2209733_0:@4
"while_simple_rnn_cell_10_2209735_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_10_2209731:@.
 while_simple_rnn_cell_10_2209733:@2
 while_simple_rnn_cell_10_2209735:@@��0while/simple_rnn_cell_10/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
0while/simple_rnn_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_10_2209731_0"while_simple_rnn_cell_10_2209733_0"while_simple_rnn_cell_10_2209735_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_2209696�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_10/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
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
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity9while/simple_rnn_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@

while/NoOpNoOp1^while/simple_rnn_cell_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_10_2209731"while_simple_rnn_cell_10_2209731_0"F
 while_simple_rnn_cell_10_2209733"while_simple_rnn_cell_10_2209733_0"F
 while_simple_rnn_cell_10_2209735"while_simple_rnn_cell_10_2209735_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2d
0while/simple_rnn_cell_10/StatefulPartitionedCall0while/simple_rnn_cell_10/StatefulPartitionedCall: 
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
:���������@:

_output_shapes
: :

_output_shapes
: 
�f
�
J__inference_sequential_18_layer_call_and_return_conditional_losses_2212982

inputsP
>simple_rnn_5_simple_rnn_cell_10_matmul_readvariableop_resource:@M
?simple_rnn_5_simple_rnn_cell_10_biasadd_readvariableop_resource:@R
@simple_rnn_5_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@4
"gru_4_read_readvariableop_resource:@`6
$gru_4_read_1_readvariableop_resource: `6
$gru_4_read_2_readvariableop_resource:`9
'dense_26_matmul_readvariableop_resource: 6
(dense_26_biasadd_readvariableop_resource:
identity��dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�gru_4/Read/ReadVariableOp�gru_4/Read_1/ReadVariableOp�gru_4/Read_2/ReadVariableOp�6simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOp�5simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOp�7simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOp�simple_rnn_5/whileH
simple_rnn_5/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_5/strided_sliceStridedSlicesimple_rnn_5/Shape:output:0)simple_rnn_5/strided_slice/stack:output:0+simple_rnn_5/strided_slice/stack_1:output:0+simple_rnn_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@�
simple_rnn_5/zeros/packedPack#simple_rnn_5/strided_slice:output:0$simple_rnn_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
simple_rnn_5/zerosFill"simple_rnn_5/zeros/packed:output:0!simple_rnn_5/zeros/Const:output:0*
T0*'
_output_shapes
:���������@p
simple_rnn_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_5/transpose	Transposeinputs$simple_rnn_5/transpose/perm:output:0*
T0*+
_output_shapes
:���������^
simple_rnn_5/Shape_1Shapesimple_rnn_5/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_5/strided_slice_1StridedSlicesimple_rnn_5/Shape_1:output:0+simple_rnn_5/strided_slice_1/stack:output:0-simple_rnn_5/strided_slice_1/stack_1:output:0-simple_rnn_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
simple_rnn_5/TensorArrayV2TensorListReserve1simple_rnn_5/TensorArrayV2/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Bsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
4simple_rnn_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_5/transpose:y:0Ksimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
"simple_rnn_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_5/strided_slice_2StridedSlicesimple_rnn_5/transpose:y:0+simple_rnn_5/strided_slice_2/stack:output:0-simple_rnn_5/strided_slice_2/stack_1:output:0-simple_rnn_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
5simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp>simple_rnn_5_simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
&simple_rnn_5/simple_rnn_cell_10/MatMulMatMul%simple_rnn_5/strided_slice_2:output:0=simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
6simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp?simple_rnn_5_simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
'simple_rnn_5/simple_rnn_cell_10/BiasAddBiasAdd0simple_rnn_5/simple_rnn_cell_10/MatMul:product:0>simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
7simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp@simple_rnn_5_simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
(simple_rnn_5/simple_rnn_cell_10/MatMul_1MatMulsimple_rnn_5/zeros:output:0?simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
#simple_rnn_5/simple_rnn_cell_10/addAddV20simple_rnn_5/simple_rnn_cell_10/BiasAdd:output:02simple_rnn_5/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
$simple_rnn_5/simple_rnn_cell_10/TanhTanh'simple_rnn_5/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@{
*simple_rnn_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
simple_rnn_5/TensorArrayV2_1TensorListReserve3simple_rnn_5/TensorArrayV2_1/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���S
simple_rnn_5/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������a
simple_rnn_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
simple_rnn_5/whileWhile(simple_rnn_5/while/loop_counter:output:0.simple_rnn_5/while/maximum_iterations:output:0simple_rnn_5/time:output:0%simple_rnn_5/TensorArrayV2_1:handle:0simple_rnn_5/zeros:output:0%simple_rnn_5/strided_slice_1:output:0Dsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0>simple_rnn_5_simple_rnn_cell_10_matmul_readvariableop_resource?simple_rnn_5_simple_rnn_cell_10_biasadd_readvariableop_resource@simple_rnn_5_simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
simple_rnn_5_while_body_2212536*+
cond#R!
simple_rnn_5_while_cond_2212535*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
=simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
/simple_rnn_5/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_5/while:output:3Fsimple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0u
"simple_rnn_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������n
$simple_rnn_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_5/strided_slice_3StridedSlice8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_5/strided_slice_3/stack:output:0-simple_rnn_5/strided_slice_3/stack_1:output:0-simple_rnn_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maskr
simple_rnn_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_5/transpose_1	Transpose8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@W
gru_4/ShapeShapesimple_rnn_5/transpose_1:y:0*
T0*
_output_shapes
:c
gru_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_4/strided_sliceStridedSlicegru_4/Shape:output:0"gru_4/strided_slice/stack:output:0$gru_4/strided_slice/stack_1:output:0$gru_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
gru_4/zeros/packedPackgru_4/strided_slice:output:0gru_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_4/zerosFillgru_4/zeros/packed:output:0gru_4/zeros/Const:output:0*
T0*'
_output_shapes
:��������� |
gru_4/Read/ReadVariableOpReadVariableOp"gru_4_read_readvariableop_resource*
_output_shapes

:@`*
dtype0f
gru_4/IdentityIdentity!gru_4/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:@`�
gru_4/Read_1/ReadVariableOpReadVariableOp$gru_4_read_1_readvariableop_resource*
_output_shapes

: `*
dtype0j
gru_4/Identity_1Identity#gru_4/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `�
gru_4/Read_2/ReadVariableOpReadVariableOp$gru_4_read_2_readvariableop_resource*
_output_shapes

:`*
dtype0j
gru_4/Identity_2Identity#gru_4/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`�
gru_4/PartitionedCallPartitionedCallsimple_rnn_5/transpose_1:y:0gru_4/zeros:output:0gru_4/Identity:output:0gru_4/Identity_1:output:0gru_4/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:��������� :��������� :��������� : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2212761�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_26/MatMulMatMulgru_4/PartitionedCall:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_26/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp^gru_4/Read/ReadVariableOp^gru_4/Read_1/ReadVariableOp^gru_4/Read_2/ReadVariableOp7^simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOp6^simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOp8^simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOp^simple_rnn_5/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp26
gru_4/Read/ReadVariableOpgru_4/Read/ReadVariableOp2:
gru_4/Read_1/ReadVariableOpgru_4/Read_1/ReadVariableOp2:
gru_4/Read_2/ReadVariableOpgru_4/Read_2/ReadVariableOp2p
6simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOp6simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOp2n
5simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOp5simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOp2r
7simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOp7simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOp2(
simple_rnn_5/whilesimple_rnn_5/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_gru_4_layer_call_and_return_conditional_losses_2213880
inputs_0.
read_readvariableop_resource:@`0
read_1_readvariableop_resource: `0
read_2_readvariableop_resource:`

identity_3��Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp=
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
valueB:�
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
value	B : s
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
:��������� p
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:@`*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:@`t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `t
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype0^

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`�
PartitionedCallPartitionedCallinputs_0zeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:��������� :��������� :��������� : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2213665i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�	
�
%__inference_signature_wrapper_2211964
simple_rnn_5_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@`
	unknown_3: `
	unknown_4:`
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_2209648o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:���������
,
_user_specified_namesimple_rnn_5_input
�
�
while_cond_2211727
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2211727___redundant_placeholder05
1while_while_cond_2211727___redundant_placeholder15
1while_while_cond_2211727___redundant_placeholder25
1while_while_cond_2211727___redundant_placeholder3
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
-: : : : :���������@: ::::: 
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
:���������@:

_output_shapes
: :

_output_shapes
:
�f
�
J__inference_sequential_18_layer_call_and_return_conditional_losses_2212494

inputsP
>simple_rnn_5_simple_rnn_cell_10_matmul_readvariableop_resource:@M
?simple_rnn_5_simple_rnn_cell_10_biasadd_readvariableop_resource:@R
@simple_rnn_5_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@4
"gru_4_read_readvariableop_resource:@`6
$gru_4_read_1_readvariableop_resource: `6
$gru_4_read_2_readvariableop_resource:`9
'dense_26_matmul_readvariableop_resource: 6
(dense_26_biasadd_readvariableop_resource:
identity��dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�gru_4/Read/ReadVariableOp�gru_4/Read_1/ReadVariableOp�gru_4/Read_2/ReadVariableOp�6simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOp�5simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOp�7simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOp�simple_rnn_5/whileH
simple_rnn_5/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_5/strided_sliceStridedSlicesimple_rnn_5/Shape:output:0)simple_rnn_5/strided_slice/stack:output:0+simple_rnn_5/strided_slice/stack_1:output:0+simple_rnn_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@�
simple_rnn_5/zeros/packedPack#simple_rnn_5/strided_slice:output:0$simple_rnn_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
simple_rnn_5/zerosFill"simple_rnn_5/zeros/packed:output:0!simple_rnn_5/zeros/Const:output:0*
T0*'
_output_shapes
:���������@p
simple_rnn_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_5/transpose	Transposeinputs$simple_rnn_5/transpose/perm:output:0*
T0*+
_output_shapes
:���������^
simple_rnn_5/Shape_1Shapesimple_rnn_5/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_5/strided_slice_1StridedSlicesimple_rnn_5/Shape_1:output:0+simple_rnn_5/strided_slice_1/stack:output:0-simple_rnn_5/strided_slice_1/stack_1:output:0-simple_rnn_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
simple_rnn_5/TensorArrayV2TensorListReserve1simple_rnn_5/TensorArrayV2/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Bsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
4simple_rnn_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_5/transpose:y:0Ksimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
"simple_rnn_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_5/strided_slice_2StridedSlicesimple_rnn_5/transpose:y:0+simple_rnn_5/strided_slice_2/stack:output:0-simple_rnn_5/strided_slice_2/stack_1:output:0-simple_rnn_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
5simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp>simple_rnn_5_simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
&simple_rnn_5/simple_rnn_cell_10/MatMulMatMul%simple_rnn_5/strided_slice_2:output:0=simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
6simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp?simple_rnn_5_simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
'simple_rnn_5/simple_rnn_cell_10/BiasAddBiasAdd0simple_rnn_5/simple_rnn_cell_10/MatMul:product:0>simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
7simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp@simple_rnn_5_simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
(simple_rnn_5/simple_rnn_cell_10/MatMul_1MatMulsimple_rnn_5/zeros:output:0?simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
#simple_rnn_5/simple_rnn_cell_10/addAddV20simple_rnn_5/simple_rnn_cell_10/BiasAdd:output:02simple_rnn_5/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
$simple_rnn_5/simple_rnn_cell_10/TanhTanh'simple_rnn_5/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@{
*simple_rnn_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
simple_rnn_5/TensorArrayV2_1TensorListReserve3simple_rnn_5/TensorArrayV2_1/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���S
simple_rnn_5/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������a
simple_rnn_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
simple_rnn_5/whileWhile(simple_rnn_5/while/loop_counter:output:0.simple_rnn_5/while/maximum_iterations:output:0simple_rnn_5/time:output:0%simple_rnn_5/TensorArrayV2_1:handle:0simple_rnn_5/zeros:output:0%simple_rnn_5/strided_slice_1:output:0Dsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0>simple_rnn_5_simple_rnn_cell_10_matmul_readvariableop_resource?simple_rnn_5_simple_rnn_cell_10_biasadd_readvariableop_resource@simple_rnn_5_simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
simple_rnn_5_while_body_2212048*+
cond#R!
simple_rnn_5_while_cond_2212047*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
=simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
/simple_rnn_5/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_5/while:output:3Fsimple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0u
"simple_rnn_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������n
$simple_rnn_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_5/strided_slice_3StridedSlice8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_5/strided_slice_3/stack:output:0-simple_rnn_5/strided_slice_3/stack_1:output:0-simple_rnn_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maskr
simple_rnn_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_5/transpose_1	Transpose8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@W
gru_4/ShapeShapesimple_rnn_5/transpose_1:y:0*
T0*
_output_shapes
:c
gru_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_4/strided_sliceStridedSlicegru_4/Shape:output:0"gru_4/strided_slice/stack:output:0$gru_4/strided_slice/stack_1:output:0$gru_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
gru_4/zeros/packedPackgru_4/strided_slice:output:0gru_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_4/zerosFillgru_4/zeros/packed:output:0gru_4/zeros/Const:output:0*
T0*'
_output_shapes
:��������� |
gru_4/Read/ReadVariableOpReadVariableOp"gru_4_read_readvariableop_resource*
_output_shapes

:@`*
dtype0f
gru_4/IdentityIdentity!gru_4/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:@`�
gru_4/Read_1/ReadVariableOpReadVariableOp$gru_4_read_1_readvariableop_resource*
_output_shapes

: `*
dtype0j
gru_4/Identity_1Identity#gru_4/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `�
gru_4/Read_2/ReadVariableOpReadVariableOp$gru_4_read_2_readvariableop_resource*
_output_shapes

:`*
dtype0j
gru_4/Identity_2Identity#gru_4/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`�
gru_4/PartitionedCallPartitionedCallsimple_rnn_5/transpose_1:y:0gru_4/zeros:output:0gru_4/Identity:output:0gru_4/Identity_1:output:0gru_4/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:��������� :��������� :��������� : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2212273�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_26/MatMulMatMulgru_4/PartitionedCall:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_26/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp^gru_4/Read/ReadVariableOp^gru_4/Read_1/ReadVariableOp^gru_4/Read_2/ReadVariableOp7^simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOp6^simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOp8^simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOp^simple_rnn_5/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp26
gru_4/Read/ReadVariableOpgru_4/Read/ReadVariableOp2:
gru_4/Read_1/ReadVariableOpgru_4/Read_1/ReadVariableOp2:
gru_4/Read_2/ReadVariableOpgru_4/Read_2/ReadVariableOp2p
6simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOp6simple_rnn_5/simple_rnn_cell_10/BiasAdd/ReadVariableOp2n
5simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOp5simple_rnn_5/simple_rnn_cell_10/MatMul/ReadVariableOp2r
7simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOp7simple_rnn_5/simple_rnn_cell_10/MatMul_1/ReadVariableOp2(
simple_rnn_5/whilesimple_rnn_5/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
��
�

<__inference___backward_gpu_gru_with_fallback_2209504_2209640
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:��������� d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:��������� `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:��������� O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:�
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:��������� q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:��������� �
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:��������� }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:��������� *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:���������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:�
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:��������� \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: �
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: �
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`�
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:���������@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:��������� e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:��������� :��������� :��������� : :��������� :��������� :��������� : ::���������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_1a12a05c-8f1c-48a5-be94-d9b5e4592f02*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2209639*
go_backwards( *

time_major( :- )
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :-)
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:���������@:1
-
+
_output_shapes
:��������� :

_output_shapes
: :!

_output_shapes	
:�I: 

_output_shapes
::-)
'
_output_shapes
:��������� :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�4
�
)__inference_gpu_gru_with_fallback_2214875

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:�IU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:��������� :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_e68a311f-c1d7-4943-85e8-55db03b37fe0*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�
�
B__inference_gru_4_layer_call_and_return_conditional_losses_2210711

inputs.
read_readvariableop_resource:@`0
read_1_readvariableop_resource: `0
read_2_readvariableop_resource:`

identity_3��Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp;
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
valueB:�
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
value	B : s
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
:��������� p
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:@`*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:@`t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `t
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype0^

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`�
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:��������� :��������� :��������� : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2210496i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�>
�
 __inference_standard_gru_2211449

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@B
ShapeShapetranspose:y:0*
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:���������`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:���������`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:��������� b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:��������� Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:��������� ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:��������� Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:��������� S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :��������� : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_2211359*
condR
while_cond_2211358*R
output_shapesA
?: : : : :��������� : : :@`:`: `:`*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:��������� ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:��������� X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_f7f2608a-40e9-4efa-aef1-50ad684e039a*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�>
�
'__forward_gpu_gru_with_fallback_2214633

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:��������� :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_dea86e7c-3720-48ac-8f3f-bb2ee451d232*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2214498_2214634*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�>
�
'__forward_gpu_gru_with_fallback_2209639

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:��������� :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_1a12a05c-8f1c-48a5-be94-d9b5e4592f02*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2209504_2209640*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�
�
B__inference_gru_4_layer_call_and_return_conditional_losses_2214258
inputs_0.
read_readvariableop_resource:@`0
read_1_readvariableop_resource: `0
read_2_readvariableop_resource:`

identity_3��Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp=
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
valueB:�
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
value	B : s
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
:��������� p
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:@`*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:@`t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `t
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype0^

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`�
PartitionedCallPartitionedCallinputs_0zeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:��������� :��������� :��������� : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2214043i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�	
�
while_cond_2210913
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2210913___redundant_placeholder05
1while_while_cond_2210913___redundant_placeholder15
1while_while_cond_2210913___redundant_placeholder25
1while_while_cond_2210913___redundant_placeholder35
1while_while_cond_2210913___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :��������� : :::::: 
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
:��������� :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
�
�
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_2209696

inputs

states0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:���������@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_namestates
�
�
B__inference_gru_4_layer_call_and_return_conditional_losses_2211219

inputs.
read_readvariableop_resource:@`0
read_1_readvariableop_resource: `0
read_2_readvariableop_resource:`

identity_3��Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp;
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
valueB:�
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
value	B : s
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
:��������� p
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:@`*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:@`t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `t
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype0^

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`�
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:��������� :��������� :��������� : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2211004i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
J__inference_sequential_18_layer_call_and_return_conditional_losses_2211912
simple_rnn_5_input&
simple_rnn_5_2211892:@"
simple_rnn_5_2211894:@&
simple_rnn_5_2211896:@@
gru_4_2211899:@`
gru_4_2211901: `
gru_4_2211903:`"
dense_26_2211906: 
dense_26_2211908:
identity�� dense_26/StatefulPartitionedCall�gru_4/StatefulPartitionedCall�$simple_rnn_5/StatefulPartitionedCall�
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_5_inputsimple_rnn_5_2211892simple_rnn_5_2211894simple_rnn_5_2211896*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2210834�
gru_4/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0gru_4_2211899gru_4_2211901gru_4_2211903*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_gru_4_layer_call_and_return_conditional_losses_2211219�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall&gru_4/StatefulPartitionedCall:output:0dense_26_2211906dense_26_2211908*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2211237x
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_26/StatefulPartitionedCall^gru_4/StatefulPartitionedCall%^simple_rnn_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2>
gru_4/StatefulPartitionedCallgru_4/StatefulPartitionedCall2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall:_ [
+
_output_shapes
:���������
,
_user_specified_namesimple_rnn_5_input
�>
�
'__forward_gpu_gru_with_fallback_2210319

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*Q
_output_shapes?
=:������������������ :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:������������������@:��������� :@`: `:`*<
api_implements*(gru_e75d9f83-e331-4427-a235-3d066454014e*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2210184_2210320*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�-
�
while_body_2214709
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:���������`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:���������`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:���������`�
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:���������`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:��������� Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:��������� t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:��������� ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:��������� o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:��������� k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:��������� U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:��������� l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:��������� P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:��������� c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:��������� h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:��������� r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:��������� "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :��������� : : :@`:`: `:`: 
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
:��������� :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
Ɠ
�

<__inference___backward_gpu_gru_with_fallback_2210184_2210320
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:��������� d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:��������� `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:��������� O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:�
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:��������� q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:��������� �
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:��������� }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :������������������ *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*T
_output_shapesB
@:������������������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :������������������@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:�
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:��������� \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: �
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: �
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`�
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :������������������@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:��������� e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:��������� :��������� :��������� : :��������� :��������� :������������������ : ::������������������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_e75d9f83-e331-4427-a235-3d066454014e*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2210319*
go_backwards( *

time_major( :- )
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :-)
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� ::6
4
_output_shapes"
 :������������������ :

_output_shapes
: :

_output_shapes
:::	6
4
_output_shapes"
 :������������������@:1
-
+
_output_shapes
:��������� :

_output_shapes
: :!

_output_shapes	
:�I: 

_output_shapes
::-)
'
_output_shapes
:��������� :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�-
�
while_body_2209337
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:���������`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:���������`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:���������`�
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:���������`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:��������� Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:��������� t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:��������� ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:��������� o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:��������� k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:��������� U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:��������� l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:��������� P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:��������� c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:��������� h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:��������� r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:��������� "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :��������� : : :@`:`: `:`: 
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
:��������� :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
Ƈ
�
#__inference__traced_restore_2215326
file_prefix2
 assignvariableop_dense_26_kernel: .
 assignvariableop_1_dense_26_bias:K
9assignvariableop_2_simple_rnn_5_simple_rnn_cell_10_kernel:@U
Cassignvariableop_3_simple_rnn_5_simple_rnn_cell_10_recurrent_kernel:@@E
7assignvariableop_4_simple_rnn_5_simple_rnn_cell_10_bias:@<
*assignvariableop_5_gru_4_gru_cell_8_kernel:@`F
4assignvariableop_6_gru_4_gru_cell_8_recurrent_kernel: `:
(assignvariableop_7_gru_4_gru_cell_8_bias:`&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: <
*assignvariableop_17_adam_dense_26_kernel_m: 6
(assignvariableop_18_adam_dense_26_bias_m:S
Aassignvariableop_19_adam_simple_rnn_5_simple_rnn_cell_10_kernel_m:@]
Kassignvariableop_20_adam_simple_rnn_5_simple_rnn_cell_10_recurrent_kernel_m:@@M
?assignvariableop_21_adam_simple_rnn_5_simple_rnn_cell_10_bias_m:@D
2assignvariableop_22_adam_gru_4_gru_cell_8_kernel_m:@`N
<assignvariableop_23_adam_gru_4_gru_cell_8_recurrent_kernel_m: `B
0assignvariableop_24_adam_gru_4_gru_cell_8_bias_m:`<
*assignvariableop_25_adam_dense_26_kernel_v: 6
(assignvariableop_26_adam_dense_26_bias_v:S
Aassignvariableop_27_adam_simple_rnn_5_simple_rnn_cell_10_kernel_v:@]
Kassignvariableop_28_adam_simple_rnn_5_simple_rnn_cell_10_recurrent_kernel_v:@@M
?assignvariableop_29_adam_simple_rnn_5_simple_rnn_cell_10_bias_v:@D
2assignvariableop_30_adam_gru_4_gru_cell_8_kernel_v:@`N
<assignvariableop_31_adam_gru_4_gru_cell_8_recurrent_kernel_v: `B
0assignvariableop_32_adam_gru_4_gru_cell_8_bias_v:`
identity_34��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*�
value�B�"B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_26_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_26_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp9assignvariableop_2_simple_rnn_5_simple_rnn_cell_10_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpCassignvariableop_3_simple_rnn_5_simple_rnn_cell_10_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp7assignvariableop_4_simple_rnn_5_simple_rnn_cell_10_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp*assignvariableop_5_gru_4_gru_cell_8_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp4assignvariableop_6_gru_4_gru_cell_8_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp(assignvariableop_7_gru_4_gru_cell_8_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_26_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_26_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpAassignvariableop_19_adam_simple_rnn_5_simple_rnn_cell_10_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpKassignvariableop_20_adam_simple_rnn_5_simple_rnn_cell_10_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp?assignvariableop_21_adam_simple_rnn_5_simple_rnn_cell_10_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp2assignvariableop_22_adam_gru_4_gru_cell_8_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp<assignvariableop_23_adam_gru_4_gru_cell_8_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp0assignvariableop_24_adam_gru_4_gru_cell_8_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_26_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_26_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpAassignvariableop_27_adam_simple_rnn_5_simple_rnn_cell_10_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpKassignvariableop_28_adam_simple_rnn_5_simple_rnn_cell_10_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp?assignvariableop_29_adam_simple_rnn_5_simple_rnn_cell_10_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp2assignvariableop_30_adam_gru_4_gru_cell_8_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp<assignvariableop_31_adam_gru_4_gru_cell_8_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp0assignvariableop_32_adam_gru_4_gru_cell_8_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
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
�>
�
 __inference_standard_gru_2214421

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@B
ShapeShapetranspose:y:0*
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:���������`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:���������`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:��������� b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:��������� Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:��������� ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:��������� Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:��������� S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :��������� : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_2214331*
condR
while_cond_2214330*R
output_shapesA
?: : : : :��������� : : :@`:`: `:`*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:��������� ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:��������� X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_dea86e7c-3720-48ac-8f3f-bb2ee451d232*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�,
�
while_body_2211728
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:@F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@��/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp�.while/simple_rnn_cell_10/MatMul/ReadVariableOp�0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@y
while/simple_rnn_cell_10/TanhTanh while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_10/Tanh:y:0*
_output_shapes
: *
element_dtype0:���M
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
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_10/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp0^while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_10/MatMul/ReadVariableOp1^while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_10_matmul_readvariableop_resource9while_simple_rnn_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2b
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_10/MatMul/ReadVariableOp.while/simple_rnn_cell_10/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 
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
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
J__inference_sequential_18_layer_call_and_return_conditional_losses_2211244

inputs&
simple_rnn_5_2210835:@"
simple_rnn_5_2210837:@&
simple_rnn_5_2210839:@@
gru_4_2211220:@`
gru_4_2211222: `
gru_4_2211224:`"
dense_26_2211238: 
dense_26_2211240:
identity�� dense_26/StatefulPartitionedCall�gru_4/StatefulPartitionedCall�$simple_rnn_5/StatefulPartitionedCall�
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_5_2210835simple_rnn_5_2210837simple_rnn_5_2210839*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2210834�
gru_4/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0gru_4_2211220gru_4_2211222gru_4_2211224*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_gru_4_layer_call_and_return_conditional_losses_2211219�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall&gru_4/StatefulPartitionedCall:output:0dense_26_2211238dense_26_2211240*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2211237x
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_26/StatefulPartitionedCall^gru_4/StatefulPartitionedCall%^simple_rnn_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2>
gru_4/StatefulPartitionedCallgru_4/StatefulPartitionedCall2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
while_cond_2212182
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2212182___redundant_placeholder05
1while_while_cond_2212182___redundant_placeholder15
1while_while_cond_2212182___redundant_placeholder25
1while_while_cond_2212182___redundant_placeholder35
1while_while_cond_2212182___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :��������� : :::::: 
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
:��������� :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
�=
�
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2211794

inputsC
1simple_rnn_cell_10_matmul_readvariableop_resource:@@
2simple_rnn_cell_10_biasadd_readvariableop_resource:@E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:@@
identity��)simple_rnn_cell_10/BiasAdd/ReadVariableOp�(simple_rnn_cell_10/MatMul/ReadVariableOp�*simple_rnn_cell_10/MatMul_1/ReadVariableOp�while;
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
valueB:�
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
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB:�
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@m
simple_rnn_cell_10/TanhTanhsimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2211728*
condR
while_cond_2211727*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_simple_rnn_5_layer_call_fn_2213026

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2211794s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_gru_4_layer_call_fn_2213469
inputs_0
unknown:@`
	unknown_0: `
	unknown_1:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_gru_4_layer_call_and_return_conditional_losses_2210322o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�

�
simple_rnn_5_while_cond_22120476
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_28
4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1O
Ksimple_rnn_5_while_simple_rnn_5_while_cond_2212047___redundant_placeholder0O
Ksimple_rnn_5_while_simple_rnn_5_while_cond_2212047___redundant_placeholder1O
Ksimple_rnn_5_while_simple_rnn_5_while_cond_2212047___redundant_placeholder2O
Ksimple_rnn_5_while_simple_rnn_5_while_cond_2212047___redundant_placeholder3
simple_rnn_5_while_identity
�
simple_rnn_5/while/LessLesssimple_rnn_5_while_placeholder4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: ::::: 
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
:���������@:

_output_shapes
: :

_output_shapes
:
�4
�
)__inference_gpu_gru_with_fallback_2211080

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:�IU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:��������� :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_be92b9e3-7179-4c2b-9a03-8f4b43aafdda*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�	
�
while_cond_2214330
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2214330___redundant_placeholder05
1while_while_cond_2214330___redundant_placeholder15
1while_while_cond_2214330___redundant_placeholder25
1while_while_cond_2214330___redundant_placeholder35
1while_while_cond_2214330___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :��������� : :::::: 
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
:��������� :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
�,
�
while_body_2213176
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:@F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@��/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp�.while/simple_rnn_cell_10/MatMul/ReadVariableOp�0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@y
while/simple_rnn_cell_10/TanhTanh while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_10/Tanh:y:0*
_output_shapes
: *
element_dtype0:���M
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
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_10/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp0^while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_10/MatMul/ReadVariableOp1^while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_10_matmul_readvariableop_resource9while_simple_rnn_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2b
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_10/MatMul/ReadVariableOp.while/simple_rnn_cell_10/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 
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
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
B__inference_gru_4_layer_call_and_return_conditional_losses_2215014

inputs.
read_readvariableop_resource:@`0
read_1_readvariableop_resource: `0
read_2_readvariableop_resource:`

identity_3��Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp;
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
valueB:�
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
value	B : s
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
:��������� p
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:@`*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:@`t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `t
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype0^

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`�
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:��������� :��������� :��������� : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2214799i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
B__inference_gru_4_layer_call_and_return_conditional_losses_2211664

inputs.
read_readvariableop_resource:@`0
read_1_readvariableop_resource: `0
read_2_readvariableop_resource:`

identity_3��Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp;
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
valueB:�
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
value	B : s
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
:��������� p
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:@`*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:@`t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `t
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype0^

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`�
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:��������� :��������� :��������� : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2211449i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�-
�
while_body_2213953
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:���������`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:���������`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split�
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:���������`�
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:���������`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:��������� Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:��������� t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:��������� ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:��������� o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:��������� k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:��������� U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:��������� l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:��������� P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:��������� c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:��������� h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:��������� r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:��������� "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :��������� : : :@`:`: `:`: 
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
:��������� :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
�>
�
 __inference_standard_gru_2212761

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@B
ShapeShapetranspose:y:0*
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:���������`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:���������`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:��������� b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:��������� Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:��������� ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:��������� Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:��������� S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :��������� : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_2212671*
condR
while_cond_2212670*R
output_shapesA
?: : : : :��������� : : :@`:`: `:`*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:��������� ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:��������� X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_39d7b2a1-7cf4-4552-ab71-1e34684b15e2*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�=
�
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2210834

inputsC
1simple_rnn_cell_10_matmul_readvariableop_resource:@@
2simple_rnn_cell_10_biasadd_readvariableop_resource:@E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:@@
identity��)simple_rnn_cell_10/BiasAdd/ReadVariableOp�(simple_rnn_cell_10/MatMul/ReadVariableOp�*simple_rnn_cell_10/MatMul_1/ReadVariableOp�while;
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
valueB:�
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
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB:�
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@m
simple_rnn_cell_10/TanhTanhsimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2210768*
condR
while_cond_2210767*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�4
�
)__inference_gpu_gru_with_fallback_2209503

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:�IU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:��������� :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_1a12a05c-8f1c-48a5-be94-d9b5e4592f02*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�	
�
/__inference_sequential_18_layer_call_fn_2212006

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@`
	unknown_3: `
	unknown_4:`
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_18_layer_call_and_return_conditional_losses_2211849o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�4
�
)__inference_gpu_gru_with_fallback_2210183

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:�IU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*Q
_output_shapes?
=:������������������ :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:������������������@:��������� :@`: `:`*<
api_implements*(gru_e75d9f83-e331-4427-a235-3d066454014e*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�
�
while_cond_2213175
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2213175___redundant_placeholder05
1while_while_cond_2213175___redundant_placeholder15
1while_while_cond_2213175___redundant_placeholder25
1while_while_cond_2213175___redundant_placeholder3
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
-: : : : :���������@: ::::: 
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
:���������@:

_output_shapes
: :

_output_shapes
:
�

�
4__inference_simple_rnn_cell_10_layer_call_fn_2215047

inputs
states_0
unknown:@
	unknown_0:@
	unknown_1:@@
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_2209696o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0
�4
�
)__inference_gpu_gru_with_fallback_2213741

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:�IU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*Q
_output_shapes?
=:������������������ :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:������������������@:��������� :@`: `:`*<
api_implements*(gru_ceb15964-ea3b-4bb7-83fb-6d4227c7beed*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�
�
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_2215078

inputs
states_00
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:���������@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0
�

�
4__inference_simple_rnn_cell_10_layer_call_fn_2215061

inputs
states_0
unknown:@
	unknown_0:@
	unknown_1:@@
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_2209816o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0
�>
�
 __inference_standard_gru_2214043

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������@B
ShapeShapetranspose:y:0*
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:���������`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:���������`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:��������� b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:��������� Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:��������� ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:��������� Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:��������� S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :��������� : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_2213953*
condR
while_cond_2213952*R
output_shapesA
?: : : : :��������� : : :@`:`: `:`*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:��������� ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:��������� X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:������������������@:��������� :@`: `:`*<
api_implements*(gru_b6541380-d387-4ad8-819e-2893f92843d6*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�
�
while_cond_2210767
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2210767___redundant_placeholder05
1while_while_cond_2210767___redundant_placeholder15
1while_while_cond_2210767___redundant_placeholder25
1while_while_cond_2210767___redundant_placeholder3
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
-: : : : :���������@: ::::: 
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
:���������@:

_output_shapes
: :

_output_shapes
:
�>
�
 __inference_standard_gru_2209427

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@B
ShapeShapetranspose:y:0*
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:���������`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:���������`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:��������� b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:��������� Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:��������� ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:��������� Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:��������� S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :��������� : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_2209337*
condR
while_cond_2209336*R
output_shapesA
?: : : : :��������� : : :@`:`: `:`*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:��������� ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:��������� X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_1a12a05c-8f1c-48a5-be94-d9b5e4592f02*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�
�
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_2215095

inputs
states_00
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:���������@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0
�4
�
)__inference_gpu_gru_with_fallback_2214497

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:�IU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:��������� :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_dea86e7c-3720-48ac-8f3f-bb2ee451d232*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�	
�
while_cond_2210405
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2210405___redundant_placeholder05
1while_while_cond_2210405___redundant_placeholder15
1while_while_cond_2210405___redundant_placeholder25
1while_while_cond_2210405___redundant_placeholder35
1while_while_cond_2210405___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :��������� : :::::: 
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
:��������� :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
��
�

<__inference___backward_gpu_gru_with_fallback_2211081_2211217
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:��������� d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:��������� `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:��������� O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:�
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:��������� q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:��������� �
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:��������� }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:��������� *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:���������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:�
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:��������� \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: �
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: �
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`�
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:���������@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:��������� e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:��������� :��������� :��������� : :��������� :��������� :��������� : ::���������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_be92b9e3-7179-4c2b-9a03-8f4b43aafdda*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2211216*
go_backwards( *

time_major( :- )
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :-)
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:���������@:1
-
+
_output_shapes
:��������� :

_output_shapes
: :!

_output_shapes	
:�I: 

_output_shapes
::-)
'
_output_shapes
:��������� :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�	
�
while_cond_2212670
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2212670___redundant_placeholder05
1while_while_cond_2212670___redundant_placeholder15
1while_while_cond_2212670___redundant_placeholder25
1while_while_cond_2212670___redundant_placeholder35
1while_while_cond_2212670___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :��������� : :::::: 
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
:��������� :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
�E
�
-sequential_18_simple_rnn_5_while_body_2209202R
Nsequential_18_simple_rnn_5_while_sequential_18_simple_rnn_5_while_loop_counterX
Tsequential_18_simple_rnn_5_while_sequential_18_simple_rnn_5_while_maximum_iterations0
,sequential_18_simple_rnn_5_while_placeholder2
.sequential_18_simple_rnn_5_while_placeholder_12
.sequential_18_simple_rnn_5_while_placeholder_2Q
Msequential_18_simple_rnn_5_while_sequential_18_simple_rnn_5_strided_slice_1_0�
�sequential_18_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_18_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0f
Tsequential_18_simple_rnn_5_while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@c
Usequential_18_simple_rnn_5_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@h
Vsequential_18_simple_rnn_5_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@-
)sequential_18_simple_rnn_5_while_identity/
+sequential_18_simple_rnn_5_while_identity_1/
+sequential_18_simple_rnn_5_while_identity_2/
+sequential_18_simple_rnn_5_while_identity_3/
+sequential_18_simple_rnn_5_while_identity_4O
Ksequential_18_simple_rnn_5_while_sequential_18_simple_rnn_5_strided_slice_1�
�sequential_18_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_18_simple_rnn_5_tensorarrayunstack_tensorlistfromtensord
Rsequential_18_simple_rnn_5_while_simple_rnn_cell_10_matmul_readvariableop_resource:@a
Ssequential_18_simple_rnn_5_while_simple_rnn_cell_10_biasadd_readvariableop_resource:@f
Tsequential_18_simple_rnn_5_while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@��Jsequential_18/simple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp�Isequential_18/simple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOp�Ksequential_18/simple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp�
Rsequential_18/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Dsequential_18/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_18_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_18_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0,sequential_18_simple_rnn_5_while_placeholder[sequential_18/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
Isequential_18/simple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpTsequential_18_simple_rnn_5_while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
:sequential_18/simple_rnn_5/while/simple_rnn_cell_10/MatMulMatMulKsequential_18/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem:item:0Qsequential_18/simple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Jsequential_18/simple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpUsequential_18_simple_rnn_5_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
;sequential_18/simple_rnn_5/while/simple_rnn_cell_10/BiasAddBiasAddDsequential_18/simple_rnn_5/while/simple_rnn_cell_10/MatMul:product:0Rsequential_18/simple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Ksequential_18/simple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpVsequential_18_simple_rnn_5_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
<sequential_18/simple_rnn_5/while/simple_rnn_cell_10/MatMul_1MatMul.sequential_18_simple_rnn_5_while_placeholder_2Ssequential_18/simple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
7sequential_18/simple_rnn_5/while/simple_rnn_cell_10/addAddV2Dsequential_18/simple_rnn_5/while/simple_rnn_cell_10/BiasAdd:output:0Fsequential_18/simple_rnn_5/while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
8sequential_18/simple_rnn_5/while/simple_rnn_cell_10/TanhTanh;sequential_18/simple_rnn_5/while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@�
Esequential_18/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_18_simple_rnn_5_while_placeholder_1,sequential_18_simple_rnn_5_while_placeholder<sequential_18/simple_rnn_5/while/simple_rnn_cell_10/Tanh:y:0*
_output_shapes
: *
element_dtype0:���h
&sequential_18/simple_rnn_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
$sequential_18/simple_rnn_5/while/addAddV2,sequential_18_simple_rnn_5_while_placeholder/sequential_18/simple_rnn_5/while/add/y:output:0*
T0*
_output_shapes
: j
(sequential_18/simple_rnn_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
&sequential_18/simple_rnn_5/while/add_1AddV2Nsequential_18_simple_rnn_5_while_sequential_18_simple_rnn_5_while_loop_counter1sequential_18/simple_rnn_5/while/add_1/y:output:0*
T0*
_output_shapes
: �
)sequential_18/simple_rnn_5/while/IdentityIdentity*sequential_18/simple_rnn_5/while/add_1:z:0&^sequential_18/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
+sequential_18/simple_rnn_5/while/Identity_1IdentityTsequential_18_simple_rnn_5_while_sequential_18_simple_rnn_5_while_maximum_iterations&^sequential_18/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
+sequential_18/simple_rnn_5/while/Identity_2Identity(sequential_18/simple_rnn_5/while/add:z:0&^sequential_18/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
+sequential_18/simple_rnn_5/while/Identity_3IdentityUsequential_18/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^sequential_18/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
+sequential_18/simple_rnn_5/while/Identity_4Identity<sequential_18/simple_rnn_5/while/simple_rnn_cell_10/Tanh:y:0&^sequential_18/simple_rnn_5/while/NoOp*
T0*'
_output_shapes
:���������@�
%sequential_18/simple_rnn_5/while/NoOpNoOpK^sequential_18/simple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpJ^sequential_18/simple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOpL^sequential_18/simple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "_
)sequential_18_simple_rnn_5_while_identity2sequential_18/simple_rnn_5/while/Identity:output:0"c
+sequential_18_simple_rnn_5_while_identity_14sequential_18/simple_rnn_5/while/Identity_1:output:0"c
+sequential_18_simple_rnn_5_while_identity_24sequential_18/simple_rnn_5/while/Identity_2:output:0"c
+sequential_18_simple_rnn_5_while_identity_34sequential_18/simple_rnn_5/while/Identity_3:output:0"c
+sequential_18_simple_rnn_5_while_identity_44sequential_18/simple_rnn_5/while/Identity_4:output:0"�
Ksequential_18_simple_rnn_5_while_sequential_18_simple_rnn_5_strided_slice_1Msequential_18_simple_rnn_5_while_sequential_18_simple_rnn_5_strided_slice_1_0"�
Ssequential_18_simple_rnn_5_while_simple_rnn_cell_10_biasadd_readvariableop_resourceUsequential_18_simple_rnn_5_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"�
Tsequential_18_simple_rnn_5_while_simple_rnn_cell_10_matmul_1_readvariableop_resourceVsequential_18_simple_rnn_5_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"�
Rsequential_18_simple_rnn_5_while_simple_rnn_cell_10_matmul_readvariableop_resourceTsequential_18_simple_rnn_5_while_simple_rnn_cell_10_matmul_readvariableop_resource_0"�
�sequential_18_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_18_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor�sequential_18_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_18_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2�
Jsequential_18/simple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpJsequential_18/simple_rnn_5/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2�
Isequential_18/simple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOpIsequential_18/simple_rnn_5/while/simple_rnn_cell_10/MatMul/ReadVariableOp2�
Ksequential_18/simple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpKsequential_18/simple_rnn_5/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 
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
:���������@:

_output_shapes
: :

_output_shapes
: 
�>
�
 __inference_standard_gru_2213665

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������@B
ShapeShapetranspose:y:0*
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:���������`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:���������`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:��������� :��������� :��������� *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:��������� b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:��������� Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:��������� ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:��������� Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:��������� I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:��������� S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:��������� V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :��������� : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_2213575*
condR
while_cond_2213574*R
output_shapesA
?: : : : :��������� : : :@`:`: `:`*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:��������� ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:��������� X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:������������������@:��������� :@`: `:`*<
api_implements*(gru_ceb15964-ea3b-4bb7-83fb-6d4227c7beed*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
��
�

<__inference___backward_gpu_gru_with_fallback_2212838_2212974
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:��������� d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:��������� `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:��������� O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:�
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:��������� q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:��������� �
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:��������� }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:��������� *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:���������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:�
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:��������� \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: �
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: �
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`�
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:���������@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:��������� e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:��������� :��������� :��������� : :��������� :��������� :��������� : ::���������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_39d7b2a1-7cf4-4552-ab71-1e34684b15e2*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2212973*
go_backwards( *

time_major( :- )
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :-)
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:���������@:1
-
+
_output_shapes
:��������� :

_output_shapes
: :!

_output_shapes	
:�I: 

_output_shapes
::-)
'
_output_shapes
:��������� :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_2213067
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2213067___redundant_placeholder05
1while_while_cond_2213067___redundant_placeholder15
1while_while_cond_2213067___redundant_placeholder25
1while_while_cond_2213067___redundant_placeholder3
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
-: : : : :���������@: ::::: 
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
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_2209708
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2209708___redundant_placeholder05
1while_while_cond_2209708___redundant_placeholder15
1while_while_cond_2209708___redundant_placeholder25
1while_while_cond_2209708___redundant_placeholder3
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
-: : : : :���������@: ::::: 
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
:���������@:

_output_shapes
: :

_output_shapes
:
��
�

<__inference___backward_gpu_gru_with_fallback_2214498_2214634
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:��������� d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:��������� `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:��������� O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:�
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:��������� q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:��������� �
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:��������� }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:��������� *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:���������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:�
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:��������� \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: �
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: �
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`�
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:���������@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:��������� e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:��������� :��������� :��������� : :��������� :��������� :��������� : ::���������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_dea86e7c-3720-48ac-8f3f-bb2ee451d232*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2214633*
go_backwards( *

time_major( :- )
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :-)
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:���������@:1
-
+
_output_shapes
:��������� :

_output_shapes
: :!

_output_shapes	
:�I: 

_output_shapes
::-)
'
_output_shapes
:��������� :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�,
�
while_body_2213392
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:@H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:@F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:@@��/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp�.while/simple_rnn_cell_10/MatMul/ReadVariableOp�0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@y
while/simple_rnn_cell_10/TanhTanh while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_10/Tanh:y:0*
_output_shapes
: *
element_dtype0:���M
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
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ~
while/Identity_4Identity!while/simple_rnn_cell_10/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp0^while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_10/MatMul/ReadVariableOp1^while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_10_matmul_readvariableop_resource9while_simple_rnn_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2b
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_10/MatMul/ReadVariableOp.while/simple_rnn_cell_10/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp0while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 
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
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
.__inference_simple_rnn_5_layer_call_fn_2212993
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2209772|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�>
�
'__forward_gpu_gru_with_fallback_2213877

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*Q
_output_shapes?
=:������������������ :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:������������������@:��������� :@`: `:`*<
api_implements*(gru_ceb15964-ea3b-4bb7-83fb-6d4227c7beed*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2213742_2213878*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�4
�
)__inference_gpu_gru_with_fallback_2212837

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:�IU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:��������� :��������� : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:��������� *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:��������� c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:��������� Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:��������� I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:���������@:��������� :@`: `:`*<
api_implements*(gru_39d7b2a1-7cf4-4552-ab71-1e34684b15e2*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
�>
�
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2213242
inputs_0C
1simple_rnn_cell_10_matmul_readvariableop_resource:@@
2simple_rnn_cell_10_biasadd_readvariableop_resource:@E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:@@
identity��)simple_rnn_cell_10/BiasAdd/ReadVariableOp�(simple_rnn_cell_10/MatMul/ReadVariableOp�*simple_rnn_cell_10/MatMul_1/ReadVariableOp�while=
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
valueB:�
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
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB:�
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:���������@m
simple_rnn_cell_10/TanhTanhsimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_2213176*
condR
while_cond_2213175*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������@�
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
��
�

<__inference___backward_gpu_gru_with_fallback_2214876_2215012
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:��������� d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:��������� `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:��������� O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:�
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:��������� q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:��������� �
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:��������� }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:�
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:��������� *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:���������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:�
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:��������� \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: �
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: �
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: �
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: �
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ �
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  �
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`�
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:���������@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:��������� e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:��������� :��������� :��������� : :��������� :��������� :��������� : ::���������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_e68a311f-c1d7-4943-85e8-55db03b37fe0*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2215011*
go_backwards( *

time_major( :- )
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :-)
'
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :1-
+
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:���������@:1
-
+
_output_shapes
:��������� :

_output_shapes
: :!

_output_shapes	
:�I: 

_output_shapes
::-)
'
_output_shapes
:��������� :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
U
simple_rnn_5_input?
$serving_default_simple_rnn_5_input:0���������<
dense_260
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
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
�
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
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
X
&0
'1
(2
)3
*4
+5
$6
%7"
trackable_list_wrapper
X
&0
'1
(2
)3
*4
+5
$6
%7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
�
1trace_0
2trace_1
3trace_2
4trace_32�
/__inference_sequential_18_layer_call_fn_2211263
/__inference_sequential_18_layer_call_fn_2211985
/__inference_sequential_18_layer_call_fn_2212006
/__inference_sequential_18_layer_call_fn_2211889�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z1trace_0z2trace_1z3trace_2z4trace_3
�
5trace_0
6trace_1
7trace_2
8trace_32�
J__inference_sequential_18_layer_call_and_return_conditional_losses_2212494
J__inference_sequential_18_layer_call_and_return_conditional_losses_2212982
J__inference_sequential_18_layer_call_and_return_conditional_losses_2211912
J__inference_sequential_18_layer_call_and_return_conditional_losses_2211935�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z5trace_0z6trace_1z7trace_2z8trace_3
�B�
"__inference__wrapped_model_2209648simple_rnn_5_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
9iter

:beta_1

;beta_2
	<decay
=learning_rate$m�%m�&m�'m�(m�)m�*m�+m�$v�%v�&v�'v�(v�)v�*v�+v�"
	optimizer
,
>serving_default"
signature_map
5
&0
'1
(2"
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

?states
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Etrace_0
Ftrace_1
Gtrace_2
Htrace_32�
.__inference_simple_rnn_5_layer_call_fn_2212993
.__inference_simple_rnn_5_layer_call_fn_2213004
.__inference_simple_rnn_5_layer_call_fn_2213015
.__inference_simple_rnn_5_layer_call_fn_2213026�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zEtrace_0zFtrace_1zGtrace_2zHtrace_3
�
Itrace_0
Jtrace_1
Ktrace_2
Ltrace_32�
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2213134
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2213242
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2213350
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2213458�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zItrace_0zJtrace_1zKtrace_2zLtrace_3
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
S_random_generator

&kernel
'recurrent_kernel
(bias"
_tf_keras_layer
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Tstates
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ztrace_0
[trace_1
\trace_2
]trace_32�
'__inference_gru_4_layer_call_fn_2213469
'__inference_gru_4_layer_call_fn_2213480
'__inference_gru_4_layer_call_fn_2213491
'__inference_gru_4_layer_call_fn_2213502�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zZtrace_0z[trace_1z\trace_2z]trace_3
�
^trace_0
_trace_1
`trace_2
atrace_32�
B__inference_gru_4_layer_call_and_return_conditional_losses_2213880
B__inference_gru_4_layer_call_and_return_conditional_losses_2214258
B__inference_gru_4_layer_call_and_return_conditional_losses_2214636
B__inference_gru_4_layer_call_and_return_conditional_losses_2215014�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z^trace_0z_trace_1z`trace_2zatrace_3
"
_generic_user_object
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
h_random_generator

)kernel
*recurrent_kernel
+bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
ntrace_02�
*__inference_dense_26_layer_call_fn_2215023�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zntrace_0
�
otrace_02�
E__inference_dense_26_layer_call_and_return_conditional_losses_2215033�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zotrace_0
!: 2dense_26/kernel
:2dense_26/bias
8:6@2&simple_rnn_5/simple_rnn_cell_10/kernel
B:@@@20simple_rnn_5/simple_rnn_cell_10/recurrent_kernel
2:0@2$simple_rnn_5/simple_rnn_cell_10/bias
):'@`2gru_4/gru_cell_8/kernel
3:1 `2!gru_4/gru_cell_8/recurrent_kernel
':%`2gru_4/gru_cell_8/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_sequential_18_layer_call_fn_2211263simple_rnn_5_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_sequential_18_layer_call_fn_2211985inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_sequential_18_layer_call_fn_2212006inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_sequential_18_layer_call_fn_2211889simple_rnn_5_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_sequential_18_layer_call_and_return_conditional_losses_2212494inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_sequential_18_layer_call_and_return_conditional_losses_2212982inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_sequential_18_layer_call_and_return_conditional_losses_2211912simple_rnn_5_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_sequential_18_layer_call_and_return_conditional_losses_2211935simple_rnn_5_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
%__inference_signature_wrapper_2211964simple_rnn_5_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
.__inference_simple_rnn_5_layer_call_fn_2212993inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_simple_rnn_5_layer_call_fn_2213004inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_simple_rnn_5_layer_call_fn_2213015inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_simple_rnn_5_layer_call_fn_2213026inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2213134inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2213242inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2213350inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2213458inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5
&0
'1
(2"
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�
wtrace_0
xtrace_12�
4__inference_simple_rnn_cell_10_layer_call_fn_2215047
4__inference_simple_rnn_cell_10_layer_call_fn_2215061�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zwtrace_0zxtrace_1
�
ytrace_0
ztrace_12�
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_2215078
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_2215095�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zytrace_0zztrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_gru_4_layer_call_fn_2213469inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_gru_4_layer_call_fn_2213480inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_gru_4_layer_call_fn_2213491inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_gru_4_layer_call_fn_2213502inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_gru_4_layer_call_and_return_conditional_losses_2213880inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_gru_4_layer_call_and_return_conditional_losses_2214258inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_gru_4_layer_call_and_return_conditional_losses_2214636inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_gru_4_layer_call_and_return_conditional_losses_2215014inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5
)0
*1
+2"
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_26_layer_call_fn_2215023inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_26_layer_call_and_return_conditional_losses_2215033inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
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
�B�
4__inference_simple_rnn_cell_10_layer_call_fn_2215047inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_simple_rnn_cell_10_layer_call_fn_2215061inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_2215078inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_2215095inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
&:$ 2Adam/dense_26/kernel/m
 :2Adam/dense_26/bias/m
=:;@2-Adam/simple_rnn_5/simple_rnn_cell_10/kernel/m
G:E@@27Adam/simple_rnn_5/simple_rnn_cell_10/recurrent_kernel/m
7:5@2+Adam/simple_rnn_5/simple_rnn_cell_10/bias/m
.:,@`2Adam/gru_4/gru_cell_8/kernel/m
8:6 `2(Adam/gru_4/gru_cell_8/recurrent_kernel/m
,:*`2Adam/gru_4/gru_cell_8/bias/m
&:$ 2Adam/dense_26/kernel/v
 :2Adam/dense_26/bias/v
=:;@2-Adam/simple_rnn_5/simple_rnn_cell_10/kernel/v
G:E@@27Adam/simple_rnn_5/simple_rnn_cell_10/recurrent_kernel/v
7:5@2+Adam/simple_rnn_5/simple_rnn_cell_10/bias/v
.:,@`2Adam/gru_4/gru_cell_8/kernel/v
8:6 `2(Adam/gru_4/gru_cell_8/recurrent_kernel/v
,:*`2Adam/gru_4/gru_cell_8/bias/v�
"__inference__wrapped_model_2209648�&(')*+$%?�<
5�2
0�-
simple_rnn_5_input���������
� "3�0
.
dense_26"�
dense_26����������
E__inference_dense_26_layer_call_and_return_conditional_losses_2215033\$%/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� }
*__inference_dense_26_layer_call_fn_2215023O$%/�,
%�"
 �
inputs��������� 
� "�����������
B__inference_gru_4_layer_call_and_return_conditional_losses_2213880})*+O�L
E�B
4�1
/�,
inputs/0������������������@

 
p 

 
� "%�"
�
0��������� 
� �
B__inference_gru_4_layer_call_and_return_conditional_losses_2214258})*+O�L
E�B
4�1
/�,
inputs/0������������������@

 
p

 
� "%�"
�
0��������� 
� �
B__inference_gru_4_layer_call_and_return_conditional_losses_2214636m)*+?�<
5�2
$�!
inputs���������@

 
p 

 
� "%�"
�
0��������� 
� �
B__inference_gru_4_layer_call_and_return_conditional_losses_2215014m)*+?�<
5�2
$�!
inputs���������@

 
p

 
� "%�"
�
0��������� 
� �
'__inference_gru_4_layer_call_fn_2213469p)*+O�L
E�B
4�1
/�,
inputs/0������������������@

 
p 

 
� "���������� �
'__inference_gru_4_layer_call_fn_2213480p)*+O�L
E�B
4�1
/�,
inputs/0������������������@

 
p

 
� "���������� �
'__inference_gru_4_layer_call_fn_2213491`)*+?�<
5�2
$�!
inputs���������@

 
p 

 
� "���������� �
'__inference_gru_4_layer_call_fn_2213502`)*+?�<
5�2
$�!
inputs���������@

 
p

 
� "���������� �
J__inference_sequential_18_layer_call_and_return_conditional_losses_2211912z&(')*+$%G�D
=�:
0�-
simple_rnn_5_input���������
p 

 
� "%�"
�
0���������
� �
J__inference_sequential_18_layer_call_and_return_conditional_losses_2211935z&(')*+$%G�D
=�:
0�-
simple_rnn_5_input���������
p

 
� "%�"
�
0���������
� �
J__inference_sequential_18_layer_call_and_return_conditional_losses_2212494n&(')*+$%;�8
1�.
$�!
inputs���������
p 

 
� "%�"
�
0���������
� �
J__inference_sequential_18_layer_call_and_return_conditional_losses_2212982n&(')*+$%;�8
1�.
$�!
inputs���������
p

 
� "%�"
�
0���������
� �
/__inference_sequential_18_layer_call_fn_2211263m&(')*+$%G�D
=�:
0�-
simple_rnn_5_input���������
p 

 
� "�����������
/__inference_sequential_18_layer_call_fn_2211889m&(')*+$%G�D
=�:
0�-
simple_rnn_5_input���������
p

 
� "�����������
/__inference_sequential_18_layer_call_fn_2211985a&(')*+$%;�8
1�.
$�!
inputs���������
p 

 
� "�����������
/__inference_sequential_18_layer_call_fn_2212006a&(')*+$%;�8
1�.
$�!
inputs���������
p

 
� "�����������
%__inference_signature_wrapper_2211964�&(')*+$%U�R
� 
K�H
F
simple_rnn_5_input0�-
simple_rnn_5_input���������"3�0
.
dense_26"�
dense_26����������
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2213134�&('O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "2�/
(�%
0������������������@
� �
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2213242�&('O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "2�/
(�%
0������������������@
� �
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2213350q&('?�<
5�2
$�!
inputs���������

 
p 

 
� ")�&
�
0���������@
� �
I__inference_simple_rnn_5_layer_call_and_return_conditional_losses_2213458q&('?�<
5�2
$�!
inputs���������

 
p

 
� ")�&
�
0���������@
� �
.__inference_simple_rnn_5_layer_call_fn_2212993}&('O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "%�"������������������@�
.__inference_simple_rnn_5_layer_call_fn_2213004}&('O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "%�"������������������@�
.__inference_simple_rnn_5_layer_call_fn_2213015d&('?�<
5�2
$�!
inputs���������

 
p 

 
� "����������@�
.__inference_simple_rnn_5_layer_call_fn_2213026d&('?�<
5�2
$�!
inputs���������

 
p

 
� "����������@�
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_2215078�&('\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������@
p 
� "R�O
H�E
�
0/0���������@
$�!
�
0/1/0���������@
� �
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_2215095�&('\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������@
p
� "R�O
H�E
�
0/0���������@
$�!
�
0/1/0���������@
� �
4__inference_simple_rnn_cell_10_layer_call_fn_2215047�&('\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������@
p 
� "D�A
�
0���������@
"�
�
1/0���������@�
4__inference_simple_rnn_cell_10_layer_call_fn_2215061�&('\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������@
p
� "D�A
�
0���������@
"�
�
1/0���������@