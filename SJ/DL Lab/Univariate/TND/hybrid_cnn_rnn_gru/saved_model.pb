�8
�"�!
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
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
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
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
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
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
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
�"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��5
�
Adam/gru_7/gru_cell_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*.
shared_nameAdam/gru_7/gru_cell_14/bias/v
�
1Adam/gru_7/gru_cell_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_7/gru_cell_14/bias/v*
_output_shapes

:`*
dtype0
�
)Adam/gru_7/gru_cell_14/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*:
shared_name+)Adam/gru_7/gru_cell_14/recurrent_kernel/v
�
=Adam/gru_7/gru_cell_14/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp)Adam/gru_7/gru_cell_14/recurrent_kernel/v*
_output_shapes

: `*
dtype0
�
Adam/gru_7/gru_cell_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@`*0
shared_name!Adam/gru_7/gru_cell_14/kernel/v
�
3Adam/gru_7/gru_cell_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_7/gru_cell_14/kernel/v*
_output_shapes

:@`*
dtype0
�
+Adam/simple_rnn_6/simple_rnn_cell_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+Adam/simple_rnn_6/simple_rnn_cell_12/bias/v
�
?Adam/simple_rnn_6/simple_rnn_cell_12/bias/v/Read/ReadVariableOpReadVariableOp+Adam/simple_rnn_6/simple_rnn_cell_12/bias/v*
_output_shapes
:@*
dtype0
�
7Adam/simple_rnn_6/simple_rnn_cell_12/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*H
shared_name97Adam/simple_rnn_6/simple_rnn_cell_12/recurrent_kernel/v
�
KAdam/simple_rnn_6/simple_rnn_cell_12/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp7Adam/simple_rnn_6/simple_rnn_cell_12/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
�
-Adam/simple_rnn_6/simple_rnn_cell_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*>
shared_name/-Adam/simple_rnn_6/simple_rnn_cell_12/kernel/v
�
AAdam/simple_rnn_6/simple_rnn_cell_12/kernel/v/Read/ReadVariableOpReadVariableOp-Adam/simple_rnn_6/simple_rnn_cell_12/kernel/v*
_output_shapes

: @*
dtype0
�
Adam/dense_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_29/bias/v
y
(Adam/dense_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_29/kernel/v
�
*Adam/dense_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_28/bias/v
y
(Adam/dense_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_28/kernel/v
�
*Adam/dense_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/v*
_output_shapes

: *
dtype0
�
Adam/conv1d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_8/bias/v
y
(Adam/conv1d_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_8/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv1d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv1d_8/kernel/v
�
*Adam/conv1d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_8/kernel/v*"
_output_shapes
: *
dtype0
�
Adam/gru_7/gru_cell_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*.
shared_nameAdam/gru_7/gru_cell_14/bias/m
�
1Adam/gru_7/gru_cell_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_7/gru_cell_14/bias/m*
_output_shapes

:`*
dtype0
�
)Adam/gru_7/gru_cell_14/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*:
shared_name+)Adam/gru_7/gru_cell_14/recurrent_kernel/m
�
=Adam/gru_7/gru_cell_14/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp)Adam/gru_7/gru_cell_14/recurrent_kernel/m*
_output_shapes

: `*
dtype0
�
Adam/gru_7/gru_cell_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@`*0
shared_name!Adam/gru_7/gru_cell_14/kernel/m
�
3Adam/gru_7/gru_cell_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_7/gru_cell_14/kernel/m*
_output_shapes

:@`*
dtype0
�
+Adam/simple_rnn_6/simple_rnn_cell_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+Adam/simple_rnn_6/simple_rnn_cell_12/bias/m
�
?Adam/simple_rnn_6/simple_rnn_cell_12/bias/m/Read/ReadVariableOpReadVariableOp+Adam/simple_rnn_6/simple_rnn_cell_12/bias/m*
_output_shapes
:@*
dtype0
�
7Adam/simple_rnn_6/simple_rnn_cell_12/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*H
shared_name97Adam/simple_rnn_6/simple_rnn_cell_12/recurrent_kernel/m
�
KAdam/simple_rnn_6/simple_rnn_cell_12/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp7Adam/simple_rnn_6/simple_rnn_cell_12/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
�
-Adam/simple_rnn_6/simple_rnn_cell_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*>
shared_name/-Adam/simple_rnn_6/simple_rnn_cell_12/kernel/m
�
AAdam/simple_rnn_6/simple_rnn_cell_12/kernel/m/Read/ReadVariableOpReadVariableOp-Adam/simple_rnn_6/simple_rnn_cell_12/kernel/m*
_output_shapes

: @*
dtype0
�
Adam/dense_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_29/bias/m
y
(Adam/dense_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_29/kernel/m
�
*Adam/dense_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_28/bias/m
y
(Adam/dense_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_28/kernel/m
�
*Adam/dense_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/m*
_output_shapes

: *
dtype0
�
Adam/conv1d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_8/bias/m
y
(Adam/conv1d_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_8/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv1d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv1d_8/kernel/m
�
*Adam/conv1d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_8/kernel/m*"
_output_shapes
: *
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
gru_7/gru_cell_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*'
shared_namegru_7/gru_cell_14/bias
�
*gru_7/gru_cell_14/bias/Read/ReadVariableOpReadVariableOpgru_7/gru_cell_14/bias*
_output_shapes

:`*
dtype0
�
"gru_7/gru_cell_14/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*3
shared_name$"gru_7/gru_cell_14/recurrent_kernel
�
6gru_7/gru_cell_14/recurrent_kernel/Read/ReadVariableOpReadVariableOp"gru_7/gru_cell_14/recurrent_kernel*
_output_shapes

: `*
dtype0
�
gru_7/gru_cell_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@`*)
shared_namegru_7/gru_cell_14/kernel
�
,gru_7/gru_cell_14/kernel/Read/ReadVariableOpReadVariableOpgru_7/gru_cell_14/kernel*
_output_shapes

:@`*
dtype0
�
$simple_rnn_6/simple_rnn_cell_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$simple_rnn_6/simple_rnn_cell_12/bias
�
8simple_rnn_6/simple_rnn_cell_12/bias/Read/ReadVariableOpReadVariableOp$simple_rnn_6/simple_rnn_cell_12/bias*
_output_shapes
:@*
dtype0
�
0simple_rnn_6/simple_rnn_cell_12/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*A
shared_name20simple_rnn_6/simple_rnn_cell_12/recurrent_kernel
�
Dsimple_rnn_6/simple_rnn_cell_12/recurrent_kernel/Read/ReadVariableOpReadVariableOp0simple_rnn_6/simple_rnn_cell_12/recurrent_kernel*
_output_shapes

:@@*
dtype0
�
&simple_rnn_6/simple_rnn_cell_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*7
shared_name(&simple_rnn_6/simple_rnn_cell_12/kernel
�
:simple_rnn_6/simple_rnn_cell_12/kernel/Read/ReadVariableOpReadVariableOp&simple_rnn_6/simple_rnn_cell_12/kernel*
_output_shapes

: @*
dtype0
r
dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_29/bias
k
!dense_29/bias/Read/ReadVariableOpReadVariableOpdense_29/bias*
_output_shapes
:*
dtype0
z
dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_29/kernel
s
#dense_29/kernel/Read/ReadVariableOpReadVariableOpdense_29/kernel*
_output_shapes

:*
dtype0
r
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_28/bias
k
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
_output_shapes
:*
dtype0
z
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_28/kernel
s
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel*
_output_shapes

: *
dtype0
r
conv1d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_8/bias
k
!conv1d_8/bias/Read/ReadVariableOpReadVariableOpconv1d_8/bias*
_output_shapes
: *
dtype0
~
conv1d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv1d_8/kernel
w
#conv1d_8/kernel/Read/ReadVariableOpReadVariableOpconv1d_8/kernel*"
_output_shapes
: *
dtype0
�
serving_default_conv1d_8_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_8_inputconv1d_8/kernelconv1d_8/bias&simple_rnn_6/simple_rnn_cell_12/kernel$simple_rnn_6/simple_rnn_cell_12/bias0simple_rnn_6/simple_rnn_cell_12/recurrent_kernelgru_7/gru_cell_14/kernel"gru_7/gru_cell_14/recurrent_kernelgru_7/gru_cell_14/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_2723542

NoOpNoOp
�`
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�_
value�_B�_ B�_
�
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses* 
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,cell
-
state_spec*
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4_random_generator
5cell
6
state_spec*
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias*
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias*
Z
0
1
G2
H3
I4
J5
K6
L7
=8
>9
E10
F11*
Z
0
1
G2
H3
I4
J5
K6
L7
=8
>9
E10
F11*
* 
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Rtrace_0
Strace_1
Ttrace_2
Utrace_3* 
6
Vtrace_0
Wtrace_1
Xtrace_2
Ytrace_3* 
* 
�
Ziter

[beta_1

\beta_2
	]decay
^learning_ratem�m�=m�>m�Em�Fm�Gm�Hm�Im�Jm�Km�Lm�v�v�=v�>v�Ev�Fv�Gv�Hv�Iv�Jv�Kv�Lv�*

_serving_default* 

0
1*

0
1*
* 
�
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

etrace_0* 

ftrace_0* 
_Y
VARIABLE_VALUEconv1d_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

ltrace_0* 

mtrace_0* 
* 
* 
* 
�
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 

strace_0* 

ttrace_0* 

G0
H1
I2*

G0
H1
I2*
* 
�

ustates
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
6
{trace_0
|trace_1
}trace_2
~trace_3* 
9
trace_0
�trace_1
�trace_2
�trace_3* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator

Gkernel
Hrecurrent_kernel
Ibias*
* 

J0
K1
L2*

J0
K1
L2*
* 
�
�states
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator

Jkernel
Krecurrent_kernel
Lbias*
* 

=0
>1*

=0
>1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_28/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_28/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

E0
F1*

E0
F1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_29/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_29/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&simple_rnn_6/simple_rnn_cell_12/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE0simple_rnn_6/simple_rnn_cell_12/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$simple_rnn_6/simple_rnn_cell_12/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEgru_7/gru_cell_14/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"gru_7/gru_cell_14/recurrent_kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEgru_7/gru_cell_14/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
1
2
3
4
5
6*

�0
�1*
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
* 
* 
* 
* 

,0*
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
G0
H1
I2*

G0
H1
I2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 

50*
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
J0
K1
L2*

J0
K1
L2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
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
VARIABLE_VALUEAdam/conv1d_8/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_8/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_28/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_28/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_29/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_29/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/simple_rnn_6/simple_rnn_cell_12/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE7Adam/simple_rnn_6/simple_rnn_cell_12/recurrent_kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE+Adam/simple_rnn_6/simple_rnn_cell_12/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gru_7/gru_cell_14/kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUE)Adam/gru_7/gru_cell_14/recurrent_kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/gru_7/gru_cell_14/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/conv1d_8/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_8/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_28/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_28/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_29/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_29/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/simple_rnn_6/simple_rnn_cell_12/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE7Adam/simple_rnn_6/simple_rnn_cell_12/recurrent_kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE+Adam/simple_rnn_6/simple_rnn_cell_12/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gru_7/gru_cell_14/kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUE)Adam/gru_7/gru_cell_14/recurrent_kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/gru_7/gru_cell_14/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_8/kernel/Read/ReadVariableOp!conv1d_8/bias/Read/ReadVariableOp#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOp#dense_29/kernel/Read/ReadVariableOp!dense_29/bias/Read/ReadVariableOp:simple_rnn_6/simple_rnn_cell_12/kernel/Read/ReadVariableOpDsimple_rnn_6/simple_rnn_cell_12/recurrent_kernel/Read/ReadVariableOp8simple_rnn_6/simple_rnn_cell_12/bias/Read/ReadVariableOp,gru_7/gru_cell_14/kernel/Read/ReadVariableOp6gru_7/gru_cell_14/recurrent_kernel/Read/ReadVariableOp*gru_7/gru_cell_14/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/conv1d_8/kernel/m/Read/ReadVariableOp(Adam/conv1d_8/bias/m/Read/ReadVariableOp*Adam/dense_28/kernel/m/Read/ReadVariableOp(Adam/dense_28/bias/m/Read/ReadVariableOp*Adam/dense_29/kernel/m/Read/ReadVariableOp(Adam/dense_29/bias/m/Read/ReadVariableOpAAdam/simple_rnn_6/simple_rnn_cell_12/kernel/m/Read/ReadVariableOpKAdam/simple_rnn_6/simple_rnn_cell_12/recurrent_kernel/m/Read/ReadVariableOp?Adam/simple_rnn_6/simple_rnn_cell_12/bias/m/Read/ReadVariableOp3Adam/gru_7/gru_cell_14/kernel/m/Read/ReadVariableOp=Adam/gru_7/gru_cell_14/recurrent_kernel/m/Read/ReadVariableOp1Adam/gru_7/gru_cell_14/bias/m/Read/ReadVariableOp*Adam/conv1d_8/kernel/v/Read/ReadVariableOp(Adam/conv1d_8/bias/v/Read/ReadVariableOp*Adam/dense_28/kernel/v/Read/ReadVariableOp(Adam/dense_28/bias/v/Read/ReadVariableOp*Adam/dense_29/kernel/v/Read/ReadVariableOp(Adam/dense_29/bias/v/Read/ReadVariableOpAAdam/simple_rnn_6/simple_rnn_cell_12/kernel/v/Read/ReadVariableOpKAdam/simple_rnn_6/simple_rnn_cell_12/recurrent_kernel/v/Read/ReadVariableOp?Adam/simple_rnn_6/simple_rnn_cell_12/bias/v/Read/ReadVariableOp3Adam/gru_7/gru_cell_14/kernel/v/Read/ReadVariableOp=Adam/gru_7/gru_cell_14/recurrent_kernel/v/Read/ReadVariableOp1Adam/gru_7/gru_cell_14/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
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
 __inference__traced_save_2726963
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_8/kernelconv1d_8/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/bias&simple_rnn_6/simple_rnn_cell_12/kernel0simple_rnn_6/simple_rnn_cell_12/recurrent_kernel$simple_rnn_6/simple_rnn_cell_12/biasgru_7/gru_cell_14/kernel"gru_7/gru_cell_14/recurrent_kernelgru_7/gru_cell_14/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv1d_8/kernel/mAdam/conv1d_8/bias/mAdam/dense_28/kernel/mAdam/dense_28/bias/mAdam/dense_29/kernel/mAdam/dense_29/bias/m-Adam/simple_rnn_6/simple_rnn_cell_12/kernel/m7Adam/simple_rnn_6/simple_rnn_cell_12/recurrent_kernel/m+Adam/simple_rnn_6/simple_rnn_cell_12/bias/mAdam/gru_7/gru_cell_14/kernel/m)Adam/gru_7/gru_cell_14/recurrent_kernel/mAdam/gru_7/gru_cell_14/bias/mAdam/conv1d_8/kernel/vAdam/conv1d_8/bias/vAdam/dense_28/kernel/vAdam/dense_28/bias/vAdam/dense_29/kernel/vAdam/dense_29/bias/v-Adam/simple_rnn_6/simple_rnn_cell_12/kernel/v7Adam/simple_rnn_6/simple_rnn_cell_12/recurrent_kernel/v+Adam/simple_rnn_6/simple_rnn_cell_12/bias/vAdam/gru_7/gru_cell_14/kernel/v)Adam/gru_7/gru_cell_14/recurrent_kernel/vAdam/gru_7/gru_cell_14/bias/v*9
Tin2
02.*
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
#__inference__traced_restore_2727108��3
�>
�
 __inference_standard_gru_2721925

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
while_body_2721835*
condR
while_cond_2721834*R
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
api_implements*(gru_e1fd2348-df97-440c-a602-69c0eee8e6d8*
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
�
b
F__inference_flatten_9_layer_call_and_return_conditional_losses_2724660

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:��������� X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
��
�

<__inference___backward_gpu_gru_with_fallback_2726567_2726703
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
:��������� *
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
7:���������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������@u
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
:���������@t

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
�:��������� :��������� :��������� : :��������� :��������� :��������� : ::���������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_72876175-a1f8-456c-a1ea-e2cc127d7930*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2726702*
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
:��������� :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:���������@:1
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
�
 __inference_standard_gru_2724397

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
:���������@B
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
while_body_2724307*
condR
while_cond_2724306*R
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_5b1fd636-c7cf-4b19-893d-fa430183f3e3*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
�
 __inference_standard_gru_2720835

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
:���������@B
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
while_body_2720745*
condR
while_cond_2720744*R
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_cb1cd7d5-d1d7-4891-9a30-91ba2b88398b*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
while_cond_2726399
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2726399___redundant_placeholder05
1while_while_cond_2726399___redundant_placeholder15
1while_while_cond_2726399___redundant_placeholder25
1while_while_cond_2726399___redundant_placeholder35
1while_while_cond_2726399___redundant_placeholder4
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
�
�
B__inference_gru_7_layer_call_and_return_conditional_losses_2721751

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
 __inference_standard_gru_2721536i

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
�
�
while_cond_2724758
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2724758___redundant_placeholder05
1while_while_cond_2724758___redundant_placeholder15
1while_while_cond_2724758___redundant_placeholder25
1while_while_cond_2724758___redundant_placeholder3
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
�
'__inference_gru_7_layer_call_fn_2725182

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
B__inference_gru_7_layer_call_and_return_conditional_losses_2722679o
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
:���������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�4
�
)__inference_gpu_gru_with_fallback_2722001

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
api_implements*(gru_e1fd2348-df97-440c-a602-69c0eee8e6d8*
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
�"
�
J__inference_sequential_20_layer_call_and_return_conditional_losses_2723470
conv1d_8_input&
conv1d_8_2723438: 
conv1d_8_2723440: &
simple_rnn_6_2723445: @"
simple_rnn_6_2723447:@&
simple_rnn_6_2723449:@@
gru_7_2723452:@`
gru_7_2723454: `
gru_7_2723456:`"
dense_28_2723459: 
dense_28_2723461:"
dense_29_2723464:
dense_29_2723466:
identity�� conv1d_8/StatefulPartitionedCall� dense_28/StatefulPartitionedCall� dense_29/StatefulPartitionedCall�gru_7/StatefulPartitionedCall�$simple_rnn_6/StatefulPartitionedCall�
 conv1d_8/StatefulPartitionedCallStatefulPartitionedCallconv1d_8_inputconv1d_8_2723438conv1d_8_2723440*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv1d_8_layer_call_and_return_conditional_losses_2722172�
flatten_9/PartitionedCallPartitionedCall)conv1d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_2722184�
repeat_vector_2/PartitionedCallPartitionedCall"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_2721074�
$simple_rnn_6/StatefulPartitionedCallStatefulPartitionedCall(repeat_vector_2/PartitionedCall:output:0simple_rnn_6_2723445simple_rnn_6_2723447simple_rnn_6_2723449*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2722294�
gru_7/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_6/StatefulPartitionedCall:output:0gru_7_2723452gru_7_2723454gru_7_2723456*
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
B__inference_gru_7_layer_call_and_return_conditional_losses_2722679�
 dense_28/StatefulPartitionedCallStatefulPartitionedCall&gru_7/StatefulPartitionedCall:output:0dense_28_2723459dense_28_2723461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_2722697�
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_2723464dense_29_2723466*
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
E__inference_dense_29_layer_call_and_return_conditional_losses_2722713x
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^conv1d_8/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall^gru_7/StatefulPartitionedCall%^simple_rnn_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2D
 conv1d_8/StatefulPartitionedCall conv1d_8/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2>
gru_7/StatefulPartitionedCallgru_7/StatefulPartitionedCall2L
$simple_rnn_6/StatefulPartitionedCall$simple_rnn_6/StatefulPartitionedCall:[ W
+
_output_shapes
:���������
(
_user_specified_nameconv1d_8_input
�
�
B__inference_gru_7_layer_call_and_return_conditional_losses_2725949
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
 __inference_standard_gru_2725734i

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
�
�
E__inference_conv1d_8_layer_call_and_return_conditional_losses_2724649

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: �
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:��������� e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
%__inference_signature_wrapper_2723542
conv1d_8_input
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@@
	unknown_4:@`
	unknown_5: `
	unknown_6:`
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv1d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_2721062o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:���������
(
_user_specified_nameconv1d_8_input
�>
�
'__forward_gpu_gru_with_fallback_2724609

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
4:��������� :��������� : :*
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_5b1fd636-c7cf-4b19-893d-fa430183f3e3*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2724474_2724610*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
�^
�
 __inference__traced_save_2726963
file_prefix.
*savev2_conv1d_8_kernel_read_readvariableop,
(savev2_conv1d_8_bias_read_readvariableop.
*savev2_dense_28_kernel_read_readvariableop,
(savev2_dense_28_bias_read_readvariableop.
*savev2_dense_29_kernel_read_readvariableop,
(savev2_dense_29_bias_read_readvariableopE
Asavev2_simple_rnn_6_simple_rnn_cell_12_kernel_read_readvariableopO
Ksavev2_simple_rnn_6_simple_rnn_cell_12_recurrent_kernel_read_readvariableopC
?savev2_simple_rnn_6_simple_rnn_cell_12_bias_read_readvariableop7
3savev2_gru_7_gru_cell_14_kernel_read_readvariableopA
=savev2_gru_7_gru_cell_14_recurrent_kernel_read_readvariableop5
1savev2_gru_7_gru_cell_14_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_conv1d_8_kernel_m_read_readvariableop3
/savev2_adam_conv1d_8_bias_m_read_readvariableop5
1savev2_adam_dense_28_kernel_m_read_readvariableop3
/savev2_adam_dense_28_bias_m_read_readvariableop5
1savev2_adam_dense_29_kernel_m_read_readvariableop3
/savev2_adam_dense_29_bias_m_read_readvariableopL
Hsavev2_adam_simple_rnn_6_simple_rnn_cell_12_kernel_m_read_readvariableopV
Rsavev2_adam_simple_rnn_6_simple_rnn_cell_12_recurrent_kernel_m_read_readvariableopJ
Fsavev2_adam_simple_rnn_6_simple_rnn_cell_12_bias_m_read_readvariableop>
:savev2_adam_gru_7_gru_cell_14_kernel_m_read_readvariableopH
Dsavev2_adam_gru_7_gru_cell_14_recurrent_kernel_m_read_readvariableop<
8savev2_adam_gru_7_gru_cell_14_bias_m_read_readvariableop5
1savev2_adam_conv1d_8_kernel_v_read_readvariableop3
/savev2_adam_conv1d_8_bias_v_read_readvariableop5
1savev2_adam_dense_28_kernel_v_read_readvariableop3
/savev2_adam_dense_28_bias_v_read_readvariableop5
1savev2_adam_dense_29_kernel_v_read_readvariableop3
/savev2_adam_dense_29_bias_v_read_readvariableopL
Hsavev2_adam_simple_rnn_6_simple_rnn_cell_12_kernel_v_read_readvariableopV
Rsavev2_adam_simple_rnn_6_simple_rnn_cell_12_recurrent_kernel_v_read_readvariableopJ
Fsavev2_adam_simple_rnn_6_simple_rnn_cell_12_bias_v_read_readvariableop>
:savev2_adam_gru_7_gru_cell_14_kernel_v_read_readvariableopH
Dsavev2_adam_gru_7_gru_cell_14_recurrent_kernel_v_read_readvariableop<
8savev2_adam_gru_7_gru_cell_14_bias_v_read_readvariableop
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*�
value�B�.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_8_kernel_read_readvariableop(savev2_conv1d_8_bias_read_readvariableop*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableop*savev2_dense_29_kernel_read_readvariableop(savev2_dense_29_bias_read_readvariableopAsavev2_simple_rnn_6_simple_rnn_cell_12_kernel_read_readvariableopKsavev2_simple_rnn_6_simple_rnn_cell_12_recurrent_kernel_read_readvariableop?savev2_simple_rnn_6_simple_rnn_cell_12_bias_read_readvariableop3savev2_gru_7_gru_cell_14_kernel_read_readvariableop=savev2_gru_7_gru_cell_14_recurrent_kernel_read_readvariableop1savev2_gru_7_gru_cell_14_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_conv1d_8_kernel_m_read_readvariableop/savev2_adam_conv1d_8_bias_m_read_readvariableop1savev2_adam_dense_28_kernel_m_read_readvariableop/savev2_adam_dense_28_bias_m_read_readvariableop1savev2_adam_dense_29_kernel_m_read_readvariableop/savev2_adam_dense_29_bias_m_read_readvariableopHsavev2_adam_simple_rnn_6_simple_rnn_cell_12_kernel_m_read_readvariableopRsavev2_adam_simple_rnn_6_simple_rnn_cell_12_recurrent_kernel_m_read_readvariableopFsavev2_adam_simple_rnn_6_simple_rnn_cell_12_bias_m_read_readvariableop:savev2_adam_gru_7_gru_cell_14_kernel_m_read_readvariableopDsavev2_adam_gru_7_gru_cell_14_recurrent_kernel_m_read_readvariableop8savev2_adam_gru_7_gru_cell_14_bias_m_read_readvariableop1savev2_adam_conv1d_8_kernel_v_read_readvariableop/savev2_adam_conv1d_8_bias_v_read_readvariableop1savev2_adam_dense_28_kernel_v_read_readvariableop/savev2_adam_dense_28_bias_v_read_readvariableop1savev2_adam_dense_29_kernel_v_read_readvariableop/savev2_adam_dense_29_bias_v_read_readvariableopHsavev2_adam_simple_rnn_6_simple_rnn_cell_12_kernel_v_read_readvariableopRsavev2_adam_simple_rnn_6_simple_rnn_cell_12_recurrent_kernel_v_read_readvariableopFsavev2_adam_simple_rnn_6_simple_rnn_cell_12_bias_v_read_readvariableop:savev2_adam_gru_7_gru_cell_14_kernel_v_read_readvariableopDsavev2_adam_gru_7_gru_cell_14_recurrent_kernel_v_read_readvariableop8savev2_adam_gru_7_gru_cell_14_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	�
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
_input_shapes�
�: : : : :::: @:@@:@:@`: `:`: : : : : : : : : : : : :::: @:@@:@:@`: `:`: : : :::: @:@@:@:@`: `:`: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: @:$ 

_output_shapes

:@@: 	

_output_shapes
:@:$
 

_output_shapes

:@`:$ 

_output_shapes

: `:$ 

_output_shapes

:`:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: @:$ 

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

:`:("$
"
_output_shapes
: : #

_output_shapes
: :$$ 

_output_shapes

: : %

_output_shapes
::$& 

_output_shapes

:: '

_output_shapes
::$( 

_output_shapes

: @:$) 

_output_shapes

:@@: *

_output_shapes
:@:$+ 

_output_shapes

:@`:$, 

_output_shapes

: `:$- 

_output_shapes

:`:.

_output_shapes
: 
�4
�
)__inference_gpu_gru_with_fallback_2723961

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
:���������@P
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
4:��������� :��������� : :*
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_f7d4556a-b6c2-4fff-93ae-297efbd138b2*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
/__inference_sequential_20_layer_call_fn_2722747
conv1d_8_input
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@@
	unknown_4:@`
	unknown_5: `
	unknown_6:`
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv1d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_20_layer_call_and_return_conditional_losses_2722720o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:���������
(
_user_specified_nameconv1d_8_input
�
h
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_2721074

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :x

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*4
_output_shapes"
 :������������������Z
stackConst*
_output_shapes
:*
dtype0*!
valueB"         p
TileTileExpandDims:output:0stack:output:0*
T0*4
_output_shapes"
 :������������������b
IdentityIdentityTile:output:0*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:������������������:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
-sequential_20_simple_rnn_6_while_cond_2720609R
Nsequential_20_simple_rnn_6_while_sequential_20_simple_rnn_6_while_loop_counterX
Tsequential_20_simple_rnn_6_while_sequential_20_simple_rnn_6_while_maximum_iterations0
,sequential_20_simple_rnn_6_while_placeholder2
.sequential_20_simple_rnn_6_while_placeholder_12
.sequential_20_simple_rnn_6_while_placeholder_2T
Psequential_20_simple_rnn_6_while_less_sequential_20_simple_rnn_6_strided_slice_1k
gsequential_20_simple_rnn_6_while_sequential_20_simple_rnn_6_while_cond_2720609___redundant_placeholder0k
gsequential_20_simple_rnn_6_while_sequential_20_simple_rnn_6_while_cond_2720609___redundant_placeholder1k
gsequential_20_simple_rnn_6_while_sequential_20_simple_rnn_6_while_cond_2720609___redundant_placeholder2k
gsequential_20_simple_rnn_6_while_sequential_20_simple_rnn_6_while_cond_2720609___redundant_placeholder3-
)sequential_20_simple_rnn_6_while_identity
�
%sequential_20/simple_rnn_6/while/LessLess,sequential_20_simple_rnn_6_while_placeholderPsequential_20_simple_rnn_6_while_less_sequential_20_simple_rnn_6_strided_slice_1*
T0*
_output_shapes
: �
)sequential_20/simple_rnn_6/while/IdentityIdentity)sequential_20/simple_rnn_6/while/Less:z:0*
T0
*
_output_shapes
: "_
)sequential_20_simple_rnn_6_while_identity2sequential_20/simple_rnn_6/while/Identity:output:0*(
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
�
�
'__inference_gru_7_layer_call_fn_2725160
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
B__inference_gru_7_layer_call_and_return_conditional_losses_2721751o
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
�=
�
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2723288

inputsC
1simple_rnn_cell_12_matmul_readvariableop_resource: @@
2simple_rnn_cell_12_biasadd_readvariableop_resource:@E
3simple_rnn_cell_12_matmul_1_readvariableop_resource:@@
identity��)simple_rnn_cell_12/BiasAdd/ReadVariableOp�(simple_rnn_cell_12/MatMul/ReadVariableOp�*simple_rnn_cell_12/MatMul_1/ReadVariableOp�while;
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
:��������� D
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
valueB"����    �
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
:��������� *
shrink_axis_mask�
(simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_12_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0�
simple_rnn_cell_12/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell_12/BiasAddBiasAdd#simple_rnn_cell_12/MatMul:product:01simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_12_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell_12/MatMul_1MatMulzeros:output:02simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell_12/addAddV2#simple_rnn_cell_12/BiasAdd:output:0%simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@m
simple_rnn_cell_12/TanhTanhsimple_rnn_cell_12/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_12_matmul_readvariableop_resource2simple_rnn_cell_12_biasadd_readvariableop_resource3simple_rnn_cell_12_matmul_1_readvariableop_resource*
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
while_body_2723222*
condR
while_cond_2723221*8
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
:���������@*
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
:���������@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp*^simple_rnn_cell_12/BiasAdd/ReadVariableOp)^simple_rnn_cell_12/MatMul/ReadVariableOp+^simple_rnn_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:��������� : : : 2V
)simple_rnn_cell_12/BiasAdd/ReadVariableOp)simple_rnn_cell_12/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_12/MatMul/ReadVariableOp(simple_rnn_cell_12/MatMul/ReadVariableOp2X
*simple_rnn_cell_12/MatMul_1/ReadVariableOp*simple_rnn_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�"
�
J__inference_sequential_20_layer_call_and_return_conditional_losses_2723379

inputs&
conv1d_8_2723347: 
conv1d_8_2723349: &
simple_rnn_6_2723354: @"
simple_rnn_6_2723356:@&
simple_rnn_6_2723358:@@
gru_7_2723361:@`
gru_7_2723363: `
gru_7_2723365:`"
dense_28_2723368: 
dense_28_2723370:"
dense_29_2723373:
dense_29_2723375:
identity�� conv1d_8/StatefulPartitionedCall� dense_28/StatefulPartitionedCall� dense_29/StatefulPartitionedCall�gru_7/StatefulPartitionedCall�$simple_rnn_6/StatefulPartitionedCall�
 conv1d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_8_2723347conv1d_8_2723349*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv1d_8_layer_call_and_return_conditional_losses_2722172�
flatten_9/PartitionedCallPartitionedCall)conv1d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_2722184�
repeat_vector_2/PartitionedCallPartitionedCall"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_2721074�
$simple_rnn_6/StatefulPartitionedCallStatefulPartitionedCall(repeat_vector_2/PartitionedCall:output:0simple_rnn_6_2723354simple_rnn_6_2723356simple_rnn_6_2723358*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2723288�
gru_7/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_6/StatefulPartitionedCall:output:0gru_7_2723361gru_7_2723363gru_7_2723365*
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
B__inference_gru_7_layer_call_and_return_conditional_losses_2723158�
 dense_28/StatefulPartitionedCallStatefulPartitionedCall&gru_7/StatefulPartitionedCall:output:0dense_28_2723368dense_28_2723370*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_2722697�
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_2723373dense_29_2723375*
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
E__inference_dense_29_layer_call_and_return_conditional_losses_2722713x
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^conv1d_8/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall^gru_7/StatefulPartitionedCall%^simple_rnn_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2D
 conv1d_8/StatefulPartitionedCall conv1d_8/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2>
gru_7/StatefulPartitionedCallgru_7/StatefulPartitionedCall2L
$simple_rnn_6/StatefulPartitionedCall$simple_rnn_6/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�4
�
)__inference_gpu_gru_with_fallback_2724473

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
:���������@P
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
4:��������� :��������� : :*
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_5b1fd636-c7cf-4b19-893d-fa430183f3e3*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
while_body_2720745
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
�
�
O__inference_simple_rnn_cell_12_layer_call_and_return_conditional_losses_2721125

inputs

states0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
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
,:��������� :���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_namestates
�
�
B__inference_gru_7_layer_call_and_return_conditional_losses_2726705

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
 __inference_standard_gru_2726490i

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
:���������@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
while_cond_2723794
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2723794___redundant_placeholder05
1while_while_cond_2723794___redundant_placeholder15
1while_while_cond_2723794___redundant_placeholder25
1while_while_cond_2723794___redundant_placeholder35
1while_while_cond_2723794___redundant_placeholder4
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
�
'__forward_gpu_gru_with_fallback_2723155

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
4:��������� :��������� : :*
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_b5d7929a-c103-4256-9d6a-9fca617f4787*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2723020_2723156*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
�
h
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_2724673

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :x

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*4
_output_shapes"
 :������������������Z
stackConst*
_output_shapes
:*
dtype0*!
valueB"         p
TileTileExpandDims:output:0stack:output:0*
T0*4
_output_shapes"
 :������������������b
IdentityIdentityTile:output:0*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:������������������:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�-
�
while_body_2721446
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
�
while_cond_2722227
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2722227___redundant_placeholder05
1while_while_cond_2722227___redundant_placeholder15
1while_while_cond_2722227___redundant_placeholder25
1while_while_cond_2722227___redundant_placeholder3
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
�
�
O__inference_simple_rnn_cell_12_layer_call_and_return_conditional_losses_2726788

inputs
states_00
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
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
,:��������� :���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0
�	
�
E__inference_dense_28_layer_call_and_return_conditional_losses_2722697

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
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
�	
�
while_cond_2725265
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2725265___redundant_placeholder05
1while_while_cond_2725265___redundant_placeholder15
1while_while_cond_2725265___redundant_placeholder25
1while_while_cond_2725265___redundant_placeholder35
1while_while_cond_2725265___redundant_placeholder4
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
while_cond_2721834
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2721834___redundant_placeholder05
1while_while_cond_2721834___redundant_placeholder15
1while_while_cond_2721834___redundant_placeholder25
1while_while_cond_2721834___redundant_placeholder35
1while_while_cond_2721834___redundant_placeholder4
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
�
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2721360

inputs,
simple_rnn_cell_12_2721285: @(
simple_rnn_cell_12_2721287:@,
simple_rnn_cell_12_2721289:@@
identity��*simple_rnn_cell_12/StatefulPartitionedCall�while;
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
 :������������������ D
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
valueB"����    �
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
:��������� *
shrink_axis_mask�
*simple_rnn_cell_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_12_2721285simple_rnn_cell_12_2721287simple_rnn_cell_12_2721289*
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
O__inference_simple_rnn_cell_12_layer_call_and_return_conditional_losses_2721245n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_12_2721285simple_rnn_cell_12_2721287simple_rnn_cell_12_2721289*
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
while_body_2721297*
condR
while_cond_2721296*8
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
NoOpNoOp+^simple_rnn_cell_12/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������ : : : 2X
*simple_rnn_cell_12/StatefulPartitionedCall*simple_rnn_cell_12/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
�
while_cond_2721296
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2721296___redundant_placeholder05
1while_while_cond_2721296___redundant_placeholder15
1while_while_cond_2721296___redundant_placeholder25
1while_while_cond_2721296___redundant_placeholder3
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
�
while_cond_2722373
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2722373___redundant_placeholder05
1while_while_cond_2722373___redundant_placeholder15
1while_while_cond_2722373___redundant_placeholder25
1while_while_cond_2722373___redundant_placeholder35
1while_while_cond_2722373___redundant_placeholder4
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
�
b
F__inference_flatten_9_layer_call_and_return_conditional_losses_2722184

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:��������� X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
Ɠ
�

<__inference___backward_gpu_gru_with_fallback_2722002_2722138
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
api_implements*(gru_e1fd2348-df97-440c-a602-69c0eee8e6d8*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2722137*
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
�
while_cond_2724306
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2724306___redundant_placeholder05
1while_while_cond_2724306___redundant_placeholder15
1while_while_cond_2724306___redundant_placeholder25
1while_while_cond_2724306___redundant_placeholder35
1while_while_cond_2724306___redundant_placeholder4
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
�
�
E__inference_conv1d_8_layer_call_and_return_conditional_losses_2722172

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: �
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:��������� e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
4__inference_simple_rnn_cell_12_layer_call_fn_2726757

inputs
states_0
unknown: @
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
O__inference_simple_rnn_cell_12_layer_call_and_return_conditional_losses_2721125o
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
,:��������� :���������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0
�>
�
'__forward_gpu_gru_with_fallback_2726702

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
4:��������� :��������� : :*
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_72876175-a1f8-456c-a1ea-e2cc127d7930*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2726567_2726703*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
while_body_2724759
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_12_matmul_readvariableop_resource_0: @H
:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_12_matmul_readvariableop_resource: @F
8while_simple_rnn_cell_12_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_12_matmul_1_readvariableop_resource:@@��/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp�.while/simple_rnn_cell_12/MatMul/ReadVariableOp�0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype0�
.while/simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_12_matmul_readvariableop_resource_0*
_output_shapes

: @*
dtype0�
while/simple_rnn_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/while/simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
 while/simple_rnn_cell_12/BiasAddBiasAdd)while/simple_rnn_cell_12/MatMul:product:07while/simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0while/simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
!while/simple_rnn_cell_12/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell_12/addAddV2)while/simple_rnn_cell_12/BiasAdd:output:0+while/simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@y
while/simple_rnn_cell_12/TanhTanh while/simple_rnn_cell_12/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_12/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_12/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp0^while/simple_rnn_cell_12/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_12/MatMul/ReadVariableOp1^while/simple_rnn_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_12_biasadd_readvariableop_resource:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_12_matmul_1_readvariableop_resource;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_12_matmul_readvariableop_resource9while_simple_rnn_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2b
/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_12/MatMul/ReadVariableOp.while/simple_rnn_cell_12/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp: 
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
�=
�
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2725149

inputsC
1simple_rnn_cell_12_matmul_readvariableop_resource: @@
2simple_rnn_cell_12_biasadd_readvariableop_resource:@E
3simple_rnn_cell_12_matmul_1_readvariableop_resource:@@
identity��)simple_rnn_cell_12/BiasAdd/ReadVariableOp�(simple_rnn_cell_12/MatMul/ReadVariableOp�*simple_rnn_cell_12/MatMul_1/ReadVariableOp�while;
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
:��������� D
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
valueB"����    �
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
:��������� *
shrink_axis_mask�
(simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_12_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0�
simple_rnn_cell_12/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell_12/BiasAddBiasAdd#simple_rnn_cell_12/MatMul:product:01simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_12_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell_12/MatMul_1MatMulzeros:output:02simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell_12/addAddV2#simple_rnn_cell_12/BiasAdd:output:0%simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@m
simple_rnn_cell_12/TanhTanhsimple_rnn_cell_12/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_12_matmul_readvariableop_resource2simple_rnn_cell_12_biasadd_readvariableop_resource3simple_rnn_cell_12_matmul_1_readvariableop_resource*
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
while_body_2725083*
condR
while_cond_2725082*8
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
:���������@*
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
:���������@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp*^simple_rnn_cell_12/BiasAdd/ReadVariableOp)^simple_rnn_cell_12/MatMul/ReadVariableOp+^simple_rnn_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:��������� : : : 2V
)simple_rnn_cell_12/BiasAdd/ReadVariableOp)simple_rnn_cell_12/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_12/MatMul/ReadVariableOp(simple_rnn_cell_12/MatMul/ReadVariableOp2X
*simple_rnn_cell_12/MatMul_1/ReadVariableOp*simple_rnn_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
/__inference_sequential_20_layer_call_fn_2723435
conv1d_8_input
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@@
	unknown_4:@`
	unknown_5: `
	unknown_6:`
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv1d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_20_layer_call_and_return_conditional_losses_2723379o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:���������
(
_user_specified_nameconv1d_8_input
�
�
'__inference_gru_7_layer_call_fn_2725171
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
B__inference_gru_7_layer_call_and_return_conditional_losses_2722140o
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
�-
�
while_body_2724307
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
/__inference_sequential_20_layer_call_fn_2723571

inputs
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@@
	unknown_4:@`
	unknown_5: `
	unknown_6:`
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_20_layer_call_and_return_conditional_losses_2722720o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�4
�
)__inference_gpu_gru_with_fallback_2726188

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
:���������@P
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
4:��������� :��������� : :*
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_4c975bea-1b37-48d1-ab33-42b4dd45878b*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
�
M
1__inference_repeat_vector_2_layer_call_fn_2724665

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_2721074m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:������������������:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
Ɠ
�

<__inference___backward_gpu_gru_with_fallback_2725433_2725569
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
api_implements*(gru_512f6324-f693-4328-96e4-47ae443c807d*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2725568*
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

�
simple_rnn_6_while_cond_27236596
2simple_rnn_6_while_simple_rnn_6_while_loop_counter<
8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations"
simple_rnn_6_while_placeholder$
 simple_rnn_6_while_placeholder_1$
 simple_rnn_6_while_placeholder_28
4simple_rnn_6_while_less_simple_rnn_6_strided_slice_1O
Ksimple_rnn_6_while_simple_rnn_6_while_cond_2723659___redundant_placeholder0O
Ksimple_rnn_6_while_simple_rnn_6_while_cond_2723659___redundant_placeholder1O
Ksimple_rnn_6_while_simple_rnn_6_while_cond_2723659___redundant_placeholder2O
Ksimple_rnn_6_while_simple_rnn_6_while_cond_2723659___redundant_placeholder3
simple_rnn_6_while_identity
�
simple_rnn_6/while/LessLesssimple_rnn_6_while_placeholder4simple_rnn_6_while_less_simple_rnn_6_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_6/while/IdentityIdentitysimple_rnn_6/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_6_while_identity$simple_rnn_6/while/Identity:output:0*(
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
�
�
O__inference_simple_rnn_cell_12_layer_call_and_return_conditional_losses_2721245

inputs

states0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
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
,:��������� :���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_namestates
܃
�

J__inference_sequential_20_layer_call_and_return_conditional_losses_2724112

inputsJ
4conv1d_8_conv1d_expanddims_1_readvariableop_resource: 6
(conv1d_8_biasadd_readvariableop_resource: P
>simple_rnn_6_simple_rnn_cell_12_matmul_readvariableop_resource: @M
?simple_rnn_6_simple_rnn_cell_12_biasadd_readvariableop_resource:@R
@simple_rnn_6_simple_rnn_cell_12_matmul_1_readvariableop_resource:@@4
"gru_7_read_readvariableop_resource:@`6
$gru_7_read_1_readvariableop_resource: `6
$gru_7_read_2_readvariableop_resource:`9
'dense_28_matmul_readvariableop_resource: 6
(dense_28_biasadd_readvariableop_resource:9
'dense_29_matmul_readvariableop_resource:6
(dense_29_biasadd_readvariableop_resource:
identity��conv1d_8/BiasAdd/ReadVariableOp�+conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp�dense_28/BiasAdd/ReadVariableOp�dense_28/MatMul/ReadVariableOp�dense_29/BiasAdd/ReadVariableOp�dense_29/MatMul/ReadVariableOp�gru_7/Read/ReadVariableOp�gru_7/Read_1/ReadVariableOp�gru_7/Read_2/ReadVariableOp�6simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOp�5simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOp�7simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOp�simple_rnn_6/whilei
conv1d_8/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d_8/Conv1D/ExpandDims
ExpandDimsinputs'conv1d_8/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
+conv1d_8/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_8_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0b
 conv1d_8/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d_8/Conv1D/ExpandDims_1
ExpandDims3conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_8/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: �
conv1d_8/Conv1DConv2D#conv1d_8/Conv1D/ExpandDims:output:0%conv1d_8/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
conv1d_8/Conv1D/SqueezeSqueezeconv1d_8/Conv1D:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

����������
conv1d_8/BiasAdd/ReadVariableOpReadVariableOp(conv1d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv1d_8/BiasAddBiasAdd conv1d_8/Conv1D/Squeeze:output:0'conv1d_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� f
conv1d_8/ReluReluconv1d_8/BiasAdd:output:0*
T0*+
_output_shapes
:��������� `
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    �
flatten_9/ReshapeReshapeconv1d_8/Relu:activations:0flatten_9/Const:output:0*
T0*'
_output_shapes
:��������� `
repeat_vector_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
repeat_vector_2/ExpandDims
ExpandDimsflatten_9/Reshape:output:0'repeat_vector_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� j
repeat_vector_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"         �
repeat_vector_2/TileTile#repeat_vector_2/ExpandDims:output:0repeat_vector_2/stack:output:0*
T0*+
_output_shapes
:��������� _
simple_rnn_6/ShapeShaperepeat_vector_2/Tile:output:0*
T0*
_output_shapes
:j
 simple_rnn_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_6/strided_sliceStridedSlicesimple_rnn_6/Shape:output:0)simple_rnn_6/strided_slice/stack:output:0+simple_rnn_6/strided_slice/stack_1:output:0+simple_rnn_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@�
simple_rnn_6/zeros/packedPack#simple_rnn_6/strided_slice:output:0$simple_rnn_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
simple_rnn_6/zerosFill"simple_rnn_6/zeros/packed:output:0!simple_rnn_6/zeros/Const:output:0*
T0*'
_output_shapes
:���������@p
simple_rnn_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_6/transpose	Transposerepeat_vector_2/Tile:output:0$simple_rnn_6/transpose/perm:output:0*
T0*+
_output_shapes
:��������� ^
simple_rnn_6/Shape_1Shapesimple_rnn_6/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_6/strided_slice_1StridedSlicesimple_rnn_6/Shape_1:output:0+simple_rnn_6/strided_slice_1/stack:output:0-simple_rnn_6/strided_slice_1/stack_1:output:0-simple_rnn_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
simple_rnn_6/TensorArrayV2TensorListReserve1simple_rnn_6/TensorArrayV2/element_shape:output:0%simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Bsimple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
4simple_rnn_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_6/transpose:y:0Ksimple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
"simple_rnn_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_6/strided_slice_2StridedSlicesimple_rnn_6/transpose:y:0+simple_rnn_6/strided_slice_2/stack:output:0-simple_rnn_6/strided_slice_2/stack_1:output:0-simple_rnn_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask�
5simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOp>simple_rnn_6_simple_rnn_cell_12_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0�
&simple_rnn_6/simple_rnn_cell_12/MatMulMatMul%simple_rnn_6/strided_slice_2:output:0=simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
6simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOp?simple_rnn_6_simple_rnn_cell_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
'simple_rnn_6/simple_rnn_cell_12/BiasAddBiasAdd0simple_rnn_6/simple_rnn_cell_12/MatMul:product:0>simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
7simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOp@simple_rnn_6_simple_rnn_cell_12_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
(simple_rnn_6/simple_rnn_cell_12/MatMul_1MatMulsimple_rnn_6/zeros:output:0?simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
#simple_rnn_6/simple_rnn_cell_12/addAddV20simple_rnn_6/simple_rnn_cell_12/BiasAdd:output:02simple_rnn_6/simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
$simple_rnn_6/simple_rnn_cell_12/TanhTanh'simple_rnn_6/simple_rnn_cell_12/add:z:0*
T0*'
_output_shapes
:���������@{
*simple_rnn_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
simple_rnn_6/TensorArrayV2_1TensorListReserve3simple_rnn_6/TensorArrayV2_1/element_shape:output:0%simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���S
simple_rnn_6/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������a
simple_rnn_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
simple_rnn_6/whileWhile(simple_rnn_6/while/loop_counter:output:0.simple_rnn_6/while/maximum_iterations:output:0simple_rnn_6/time:output:0%simple_rnn_6/TensorArrayV2_1:handle:0simple_rnn_6/zeros:output:0%simple_rnn_6/strided_slice_1:output:0Dsimple_rnn_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0>simple_rnn_6_simple_rnn_cell_12_matmul_readvariableop_resource?simple_rnn_6_simple_rnn_cell_12_biasadd_readvariableop_resource@simple_rnn_6_simple_rnn_cell_12_matmul_1_readvariableop_resource*
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
simple_rnn_6_while_body_2723660*+
cond#R!
simple_rnn_6_while_cond_2723659*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
=simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
/simple_rnn_6/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_6/while:output:3Fsimple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0u
"simple_rnn_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������n
$simple_rnn_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_6/strided_slice_3StridedSlice8simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_6/strided_slice_3/stack:output:0-simple_rnn_6/strided_slice_3/stack_1:output:0-simple_rnn_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maskr
simple_rnn_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_6/transpose_1	Transpose8simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@W
gru_7/ShapeShapesimple_rnn_6/transpose_1:y:0*
T0*
_output_shapes
:c
gru_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_7/strided_sliceStridedSlicegru_7/Shape:output:0"gru_7/strided_slice/stack:output:0$gru_7/strided_slice/stack_1:output:0$gru_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
gru_7/zeros/packedPackgru_7/strided_slice:output:0gru_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_7/zerosFillgru_7/zeros/packed:output:0gru_7/zeros/Const:output:0*
T0*'
_output_shapes
:��������� |
gru_7/Read/ReadVariableOpReadVariableOp"gru_7_read_readvariableop_resource*
_output_shapes

:@`*
dtype0f
gru_7/IdentityIdentity!gru_7/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:@`�
gru_7/Read_1/ReadVariableOpReadVariableOp$gru_7_read_1_readvariableop_resource*
_output_shapes

: `*
dtype0j
gru_7/Identity_1Identity#gru_7/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `�
gru_7/Read_2/ReadVariableOpReadVariableOp$gru_7_read_2_readvariableop_resource*
_output_shapes

:`*
dtype0j
gru_7/Identity_2Identity#gru_7/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`�
gru_7/PartitionedCallPartitionedCallsimple_rnn_6/transpose_1:y:0gru_7/zeros:output:0gru_7/Identity:output:0gru_7/Identity_1:output:0gru_7/Identity_2:output:0*
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
 __inference_standard_gru_2723885�
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_28/MatMulMatMulgru_7/PartitionedCall:output:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_29/MatMulMatMuldense_28/BiasAdd:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_29/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^conv1d_8/BiasAdd/ReadVariableOp,^conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp^gru_7/Read/ReadVariableOp^gru_7/Read_1/ReadVariableOp^gru_7/Read_2/ReadVariableOp7^simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOp6^simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOp8^simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOp^simple_rnn_6/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2B
conv1d_8/BiasAdd/ReadVariableOpconv1d_8/BiasAdd/ReadVariableOp2Z
+conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp26
gru_7/Read/ReadVariableOpgru_7/Read/ReadVariableOp2:
gru_7/Read_1/ReadVariableOpgru_7/Read_1/ReadVariableOp2:
gru_7/Read_2/ReadVariableOpgru_7/Read_2/ReadVariableOp2p
6simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOp6simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOp2n
5simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOp5simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOp2r
7simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOp7simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOp2(
simple_rnn_6/whilesimple_rnn_6/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�4
�
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2721201

inputs,
simple_rnn_cell_12_2721126: @(
simple_rnn_cell_12_2721128:@,
simple_rnn_cell_12_2721130:@@
identity��*simple_rnn_cell_12/StatefulPartitionedCall�while;
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
 :������������������ D
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
valueB"����    �
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
:��������� *
shrink_axis_mask�
*simple_rnn_cell_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_12_2721126simple_rnn_cell_12_2721128simple_rnn_cell_12_2721130*
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
O__inference_simple_rnn_cell_12_layer_call_and_return_conditional_losses_2721125n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_12_2721126simple_rnn_cell_12_2721128simple_rnn_cell_12_2721130*
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
while_body_2721138*
condR
while_cond_2721137*8
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
NoOpNoOp+^simple_rnn_cell_12/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������ : : : 2X
*simple_rnn_cell_12/StatefulPartitionedCall*simple_rnn_cell_12/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�8
�
simple_rnn_6_while_body_27236606
2simple_rnn_6_while_simple_rnn_6_while_loop_counter<
8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations"
simple_rnn_6_while_placeholder$
 simple_rnn_6_while_placeholder_1$
 simple_rnn_6_while_placeholder_25
1simple_rnn_6_while_simple_rnn_6_strided_slice_1_0q
msimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0X
Fsimple_rnn_6_while_simple_rnn_cell_12_matmul_readvariableop_resource_0: @U
Gsimple_rnn_6_while_simple_rnn_cell_12_biasadd_readvariableop_resource_0:@Z
Hsimple_rnn_6_while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0:@@
simple_rnn_6_while_identity!
simple_rnn_6_while_identity_1!
simple_rnn_6_while_identity_2!
simple_rnn_6_while_identity_3!
simple_rnn_6_while_identity_43
/simple_rnn_6_while_simple_rnn_6_strided_slice_1o
ksimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensorV
Dsimple_rnn_6_while_simple_rnn_cell_12_matmul_readvariableop_resource: @S
Esimple_rnn_6_while_simple_rnn_cell_12_biasadd_readvariableop_resource:@X
Fsimple_rnn_6_while_simple_rnn_cell_12_matmul_1_readvariableop_resource:@@��<simple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp�;simple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOp�=simple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOp�
Dsimple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
6simple_rnn_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_6_while_placeholderMsimple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype0�
;simple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOpFsimple_rnn_6_while_simple_rnn_cell_12_matmul_readvariableop_resource_0*
_output_shapes

: @*
dtype0�
,simple_rnn_6/while/simple_rnn_cell_12/MatMulMatMul=simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem:item:0Csimple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<simple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOpGsimple_rnn_6_while_simple_rnn_cell_12_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
-simple_rnn_6/while/simple_rnn_cell_12/BiasAddBiasAdd6simple_rnn_6/while/simple_rnn_cell_12/MatMul:product:0Dsimple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
=simple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOpHsimple_rnn_6_while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
.simple_rnn_6/while/simple_rnn_cell_12/MatMul_1MatMul simple_rnn_6_while_placeholder_2Esimple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)simple_rnn_6/while/simple_rnn_cell_12/addAddV26simple_rnn_6/while/simple_rnn_cell_12/BiasAdd:output:08simple_rnn_6/while/simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
*simple_rnn_6/while/simple_rnn_cell_12/TanhTanh-simple_rnn_6/while/simple_rnn_cell_12/add:z:0*
T0*'
_output_shapes
:���������@�
7simple_rnn_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_6_while_placeholder_1simple_rnn_6_while_placeholder.simple_rnn_6/while/simple_rnn_cell_12/Tanh:y:0*
_output_shapes
: *
element_dtype0:���Z
simple_rnn_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_6/while/addAddV2simple_rnn_6_while_placeholder!simple_rnn_6/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_6/while/add_1AddV22simple_rnn_6_while_simple_rnn_6_while_loop_counter#simple_rnn_6/while/add_1/y:output:0*
T0*
_output_shapes
: �
simple_rnn_6/while/IdentityIdentitysimple_rnn_6/while/add_1:z:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_6/while/Identity_1Identity8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_6/while/Identity_2Identitysimple_rnn_6/while/add:z:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_6/while/Identity_3IdentityGsimple_rnn_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_6/while/Identity_4Identity.simple_rnn_6/while/simple_rnn_cell_12/Tanh:y:0^simple_rnn_6/while/NoOp*
T0*'
_output_shapes
:���������@�
simple_rnn_6/while/NoOpNoOp=^simple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp<^simple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOp>^simple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_6_while_identity$simple_rnn_6/while/Identity:output:0"G
simple_rnn_6_while_identity_1&simple_rnn_6/while/Identity_1:output:0"G
simple_rnn_6_while_identity_2&simple_rnn_6/while/Identity_2:output:0"G
simple_rnn_6_while_identity_3&simple_rnn_6/while/Identity_3:output:0"G
simple_rnn_6_while_identity_4&simple_rnn_6/while/Identity_4:output:0"d
/simple_rnn_6_while_simple_rnn_6_strided_slice_11simple_rnn_6_while_simple_rnn_6_strided_slice_1_0"�
Esimple_rnn_6_while_simple_rnn_cell_12_biasadd_readvariableop_resourceGsimple_rnn_6_while_simple_rnn_cell_12_biasadd_readvariableop_resource_0"�
Fsimple_rnn_6_while_simple_rnn_cell_12_matmul_1_readvariableop_resourceHsimple_rnn_6_while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0"�
Dsimple_rnn_6_while_simple_rnn_cell_12_matmul_readvariableop_resourceFsimple_rnn_6_while_simple_rnn_cell_12_matmul_readvariableop_resource_0"�
ksimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensormsimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2|
<simple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp<simple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp2z
;simple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOp;simple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOp2~
=simple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOp=simple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOp: 
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
�
�
#__inference__traced_restore_2727108
file_prefix6
 assignvariableop_conv1d_8_kernel: .
 assignvariableop_1_conv1d_8_bias: 4
"assignvariableop_2_dense_28_kernel: .
 assignvariableop_3_dense_28_bias:4
"assignvariableop_4_dense_29_kernel:.
 assignvariableop_5_dense_29_bias:K
9assignvariableop_6_simple_rnn_6_simple_rnn_cell_12_kernel: @U
Cassignvariableop_7_simple_rnn_6_simple_rnn_cell_12_recurrent_kernel:@@E
7assignvariableop_8_simple_rnn_6_simple_rnn_cell_12_bias:@=
+assignvariableop_9_gru_7_gru_cell_14_kernel:@`H
6assignvariableop_10_gru_7_gru_cell_14_recurrent_kernel: `<
*assignvariableop_11_gru_7_gru_cell_14_bias:`'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: #
assignvariableop_19_total: #
assignvariableop_20_count: @
*assignvariableop_21_adam_conv1d_8_kernel_m: 6
(assignvariableop_22_adam_conv1d_8_bias_m: <
*assignvariableop_23_adam_dense_28_kernel_m: 6
(assignvariableop_24_adam_dense_28_bias_m:<
*assignvariableop_25_adam_dense_29_kernel_m:6
(assignvariableop_26_adam_dense_29_bias_m:S
Aassignvariableop_27_adam_simple_rnn_6_simple_rnn_cell_12_kernel_m: @]
Kassignvariableop_28_adam_simple_rnn_6_simple_rnn_cell_12_recurrent_kernel_m:@@M
?assignvariableop_29_adam_simple_rnn_6_simple_rnn_cell_12_bias_m:@E
3assignvariableop_30_adam_gru_7_gru_cell_14_kernel_m:@`O
=assignvariableop_31_adam_gru_7_gru_cell_14_recurrent_kernel_m: `C
1assignvariableop_32_adam_gru_7_gru_cell_14_bias_m:`@
*assignvariableop_33_adam_conv1d_8_kernel_v: 6
(assignvariableop_34_adam_conv1d_8_bias_v: <
*assignvariableop_35_adam_dense_28_kernel_v: 6
(assignvariableop_36_adam_dense_28_bias_v:<
*assignvariableop_37_adam_dense_29_kernel_v:6
(assignvariableop_38_adam_dense_29_bias_v:S
Aassignvariableop_39_adam_simple_rnn_6_simple_rnn_cell_12_kernel_v: @]
Kassignvariableop_40_adam_simple_rnn_6_simple_rnn_cell_12_recurrent_kernel_v:@@M
?assignvariableop_41_adam_simple_rnn_6_simple_rnn_cell_12_bias_v:@E
3assignvariableop_42_adam_gru_7_gru_cell_14_kernel_v:@`O
=assignvariableop_43_adam_gru_7_gru_cell_14_recurrent_kernel_v: `C
1assignvariableop_44_adam_gru_7_gru_cell_14_bias_v:`
identity_46��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*�
value�B�.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_conv1d_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_28_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_28_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_29_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_29_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp9assignvariableop_6_simple_rnn_6_simple_rnn_cell_12_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpCassignvariableop_7_simple_rnn_6_simple_rnn_cell_12_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp7assignvariableop_8_simple_rnn_6_simple_rnn_cell_12_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp+assignvariableop_9_gru_7_gru_cell_14_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp6assignvariableop_10_gru_7_gru_cell_14_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp*assignvariableop_11_gru_7_gru_cell_14_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv1d_8_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv1d_8_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_28_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_28_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_29_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_29_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpAassignvariableop_27_adam_simple_rnn_6_simple_rnn_cell_12_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpKassignvariableop_28_adam_simple_rnn_6_simple_rnn_cell_12_recurrent_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp?assignvariableop_29_adam_simple_rnn_6_simple_rnn_cell_12_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp3assignvariableop_30_adam_gru_7_gru_cell_14_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp=assignvariableop_31_adam_gru_7_gru_cell_14_recurrent_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp1assignvariableop_32_adam_gru_7_gru_cell_14_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv1d_8_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv1d_8_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_28_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_28_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_29_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_29_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpAassignvariableop_39_adam_simple_rnn_6_simple_rnn_cell_12_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpKassignvariableop_40_adam_simple_rnn_6_simple_rnn_cell_12_recurrent_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp?assignvariableop_41_adam_simple_rnn_6_simple_rnn_cell_12_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp3assignvariableop_42_adam_gru_7_gru_cell_14_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp=assignvariableop_43_adam_gru_7_gru_cell_14_recurrent_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp1assignvariableop_44_adam_gru_7_gru_cell_14_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
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
�!
�
while_body_2721138
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_12_2721160_0: @0
"while_simple_rnn_cell_12_2721162_0:@4
"while_simple_rnn_cell_12_2721164_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_12_2721160: @.
 while_simple_rnn_cell_12_2721162:@2
 while_simple_rnn_cell_12_2721164:@@��0while/simple_rnn_cell_12/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype0�
0while/simple_rnn_cell_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_12_2721160_0"while_simple_rnn_cell_12_2721162_0"while_simple_rnn_cell_12_2721164_0*
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
O__inference_simple_rnn_cell_12_layer_call_and_return_conditional_losses_2721125�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_12/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity9while/simple_rnn_cell_12/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@

while/NoOpNoOp1^while/simple_rnn_cell_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_12_2721160"while_simple_rnn_cell_12_2721160_0"F
 while_simple_rnn_cell_12_2721162"while_simple_rnn_cell_12_2721162_0"F
 while_simple_rnn_cell_12_2721164"while_simple_rnn_cell_12_2721164_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2d
0while/simple_rnn_cell_12/StatefulPartitionedCall0while/simple_rnn_cell_12/StatefulPartitionedCall: 
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
�4
�
)__inference_gpu_gru_with_fallback_2725432

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
api_implements*(gru_512f6324-f693-4328-96e4-47ae443c807d*
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
��
�

<__inference___backward_gpu_gru_with_fallback_2724474_2724610
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
:��������� *
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
7:���������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������@u
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
:���������@t

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
�:��������� :��������� :��������� : :��������� :��������� :��������� : ::���������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_5b1fd636-c7cf-4b19-893d-fa430183f3e3*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2724609*
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
:��������� :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:���������@:1
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
��
�
"__inference__wrapped_model_2721062
conv1d_8_inputX
Bsequential_20_conv1d_8_conv1d_expanddims_1_readvariableop_resource: D
6sequential_20_conv1d_8_biasadd_readvariableop_resource: ^
Lsequential_20_simple_rnn_6_simple_rnn_cell_12_matmul_readvariableop_resource: @[
Msequential_20_simple_rnn_6_simple_rnn_cell_12_biasadd_readvariableop_resource:@`
Nsequential_20_simple_rnn_6_simple_rnn_cell_12_matmul_1_readvariableop_resource:@@B
0sequential_20_gru_7_read_readvariableop_resource:@`D
2sequential_20_gru_7_read_1_readvariableop_resource: `D
2sequential_20_gru_7_read_2_readvariableop_resource:`G
5sequential_20_dense_28_matmul_readvariableop_resource: D
6sequential_20_dense_28_biasadd_readvariableop_resource:G
5sequential_20_dense_29_matmul_readvariableop_resource:D
6sequential_20_dense_29_biasadd_readvariableop_resource:
identity��-sequential_20/conv1d_8/BiasAdd/ReadVariableOp�9sequential_20/conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp�-sequential_20/dense_28/BiasAdd/ReadVariableOp�,sequential_20/dense_28/MatMul/ReadVariableOp�-sequential_20/dense_29/BiasAdd/ReadVariableOp�,sequential_20/dense_29/MatMul/ReadVariableOp�'sequential_20/gru_7/Read/ReadVariableOp�)sequential_20/gru_7/Read_1/ReadVariableOp�)sequential_20/gru_7/Read_2/ReadVariableOp�Dsequential_20/simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOp�Csequential_20/simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOp�Esequential_20/simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOp� sequential_20/simple_rnn_6/whilew
,sequential_20/conv1d_8/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
(sequential_20/conv1d_8/Conv1D/ExpandDims
ExpandDimsconv1d_8_input5sequential_20/conv1d_8/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
9sequential_20/conv1d_8/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpBsequential_20_conv1d_8_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0p
.sequential_20/conv1d_8/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_20/conv1d_8/Conv1D/ExpandDims_1
ExpandDimsAsequential_20/conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp:value:07sequential_20/conv1d_8/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: �
sequential_20/conv1d_8/Conv1DConv2D1sequential_20/conv1d_8/Conv1D/ExpandDims:output:03sequential_20/conv1d_8/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
%sequential_20/conv1d_8/Conv1D/SqueezeSqueeze&sequential_20/conv1d_8/Conv1D:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

����������
-sequential_20/conv1d_8/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_conv1d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_20/conv1d_8/BiasAddBiasAdd.sequential_20/conv1d_8/Conv1D/Squeeze:output:05sequential_20/conv1d_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� �
sequential_20/conv1d_8/ReluRelu'sequential_20/conv1d_8/BiasAdd:output:0*
T0*+
_output_shapes
:��������� n
sequential_20/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    �
sequential_20/flatten_9/ReshapeReshape)sequential_20/conv1d_8/Relu:activations:0&sequential_20/flatten_9/Const:output:0*
T0*'
_output_shapes
:��������� n
,sequential_20/repeat_vector_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
(sequential_20/repeat_vector_2/ExpandDims
ExpandDims(sequential_20/flatten_9/Reshape:output:05sequential_20/repeat_vector_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� x
#sequential_20/repeat_vector_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"         �
"sequential_20/repeat_vector_2/TileTile1sequential_20/repeat_vector_2/ExpandDims:output:0,sequential_20/repeat_vector_2/stack:output:0*
T0*+
_output_shapes
:��������� {
 sequential_20/simple_rnn_6/ShapeShape+sequential_20/repeat_vector_2/Tile:output:0*
T0*
_output_shapes
:x
.sequential_20/simple_rnn_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential_20/simple_rnn_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential_20/simple_rnn_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(sequential_20/simple_rnn_6/strided_sliceStridedSlice)sequential_20/simple_rnn_6/Shape:output:07sequential_20/simple_rnn_6/strided_slice/stack:output:09sequential_20/simple_rnn_6/strided_slice/stack_1:output:09sequential_20/simple_rnn_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_20/simple_rnn_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@�
'sequential_20/simple_rnn_6/zeros/packedPack1sequential_20/simple_rnn_6/strided_slice:output:02sequential_20/simple_rnn_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&sequential_20/simple_rnn_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
 sequential_20/simple_rnn_6/zerosFill0sequential_20/simple_rnn_6/zeros/packed:output:0/sequential_20/simple_rnn_6/zeros/Const:output:0*
T0*'
_output_shapes
:���������@~
)sequential_20/simple_rnn_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
$sequential_20/simple_rnn_6/transpose	Transpose+sequential_20/repeat_vector_2/Tile:output:02sequential_20/simple_rnn_6/transpose/perm:output:0*
T0*+
_output_shapes
:��������� z
"sequential_20/simple_rnn_6/Shape_1Shape(sequential_20/simple_rnn_6/transpose:y:0*
T0*
_output_shapes
:z
0sequential_20/simple_rnn_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_20/simple_rnn_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_20/simple_rnn_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*sequential_20/simple_rnn_6/strided_slice_1StridedSlice+sequential_20/simple_rnn_6/Shape_1:output:09sequential_20/simple_rnn_6/strided_slice_1/stack:output:0;sequential_20/simple_rnn_6/strided_slice_1/stack_1:output:0;sequential_20/simple_rnn_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
6sequential_20/simple_rnn_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
(sequential_20/simple_rnn_6/TensorArrayV2TensorListReserve?sequential_20/simple_rnn_6/TensorArrayV2/element_shape:output:03sequential_20/simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Psequential_20/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
Bsequential_20/simple_rnn_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_20/simple_rnn_6/transpose:y:0Ysequential_20/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���z
0sequential_20/simple_rnn_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_20/simple_rnn_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_20/simple_rnn_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*sequential_20/simple_rnn_6/strided_slice_2StridedSlice(sequential_20/simple_rnn_6/transpose:y:09sequential_20/simple_rnn_6/strided_slice_2/stack:output:0;sequential_20/simple_rnn_6/strided_slice_2/stack_1:output:0;sequential_20/simple_rnn_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask�
Csequential_20/simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOpLsequential_20_simple_rnn_6_simple_rnn_cell_12_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0�
4sequential_20/simple_rnn_6/simple_rnn_cell_12/MatMulMatMul3sequential_20/simple_rnn_6/strided_slice_2:output:0Ksequential_20/simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Dsequential_20/simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOpMsequential_20_simple_rnn_6_simple_rnn_cell_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
5sequential_20/simple_rnn_6/simple_rnn_cell_12/BiasAddBiasAdd>sequential_20/simple_rnn_6/simple_rnn_cell_12/MatMul:product:0Lsequential_20/simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Esequential_20/simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOpNsequential_20_simple_rnn_6_simple_rnn_cell_12_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
6sequential_20/simple_rnn_6/simple_rnn_cell_12/MatMul_1MatMul)sequential_20/simple_rnn_6/zeros:output:0Msequential_20/simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
1sequential_20/simple_rnn_6/simple_rnn_cell_12/addAddV2>sequential_20/simple_rnn_6/simple_rnn_cell_12/BiasAdd:output:0@sequential_20/simple_rnn_6/simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
2sequential_20/simple_rnn_6/simple_rnn_cell_12/TanhTanh5sequential_20/simple_rnn_6/simple_rnn_cell_12/add:z:0*
T0*'
_output_shapes
:���������@�
8sequential_20/simple_rnn_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
*sequential_20/simple_rnn_6/TensorArrayV2_1TensorListReserveAsequential_20/simple_rnn_6/TensorArrayV2_1/element_shape:output:03sequential_20/simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���a
sequential_20/simple_rnn_6/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3sequential_20/simple_rnn_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������o
-sequential_20/simple_rnn_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
 sequential_20/simple_rnn_6/whileWhile6sequential_20/simple_rnn_6/while/loop_counter:output:0<sequential_20/simple_rnn_6/while/maximum_iterations:output:0(sequential_20/simple_rnn_6/time:output:03sequential_20/simple_rnn_6/TensorArrayV2_1:handle:0)sequential_20/simple_rnn_6/zeros:output:03sequential_20/simple_rnn_6/strided_slice_1:output:0Rsequential_20/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0Lsequential_20_simple_rnn_6_simple_rnn_cell_12_matmul_readvariableop_resourceMsequential_20_simple_rnn_6_simple_rnn_cell_12_biasadd_readvariableop_resourceNsequential_20_simple_rnn_6_simple_rnn_cell_12_matmul_1_readvariableop_resource*
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
-sequential_20_simple_rnn_6_while_body_2720610*9
cond1R/
-sequential_20_simple_rnn_6_while_cond_2720609*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
Ksequential_20/simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
=sequential_20/simple_rnn_6/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_20/simple_rnn_6/while:output:3Tsequential_20/simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0�
0sequential_20/simple_rnn_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������|
2sequential_20/simple_rnn_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2sequential_20/simple_rnn_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*sequential_20/simple_rnn_6/strided_slice_3StridedSliceFsequential_20/simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:09sequential_20/simple_rnn_6/strided_slice_3/stack:output:0;sequential_20/simple_rnn_6/strided_slice_3/stack_1:output:0;sequential_20/simple_rnn_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask�
+sequential_20/simple_rnn_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
&sequential_20/simple_rnn_6/transpose_1	TransposeFsequential_20/simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:04sequential_20/simple_rnn_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@s
sequential_20/gru_7/ShapeShape*sequential_20/simple_rnn_6/transpose_1:y:0*
T0*
_output_shapes
:q
'sequential_20/gru_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_20/gru_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_20/gru_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!sequential_20/gru_7/strided_sliceStridedSlice"sequential_20/gru_7/Shape:output:00sequential_20/gru_7/strided_slice/stack:output:02sequential_20/gru_7/strided_slice/stack_1:output:02sequential_20/gru_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_20/gru_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
 sequential_20/gru_7/zeros/packedPack*sequential_20/gru_7/strided_slice:output:0+sequential_20/gru_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_20/gru_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_20/gru_7/zerosFill)sequential_20/gru_7/zeros/packed:output:0(sequential_20/gru_7/zeros/Const:output:0*
T0*'
_output_shapes
:��������� �
'sequential_20/gru_7/Read/ReadVariableOpReadVariableOp0sequential_20_gru_7_read_readvariableop_resource*
_output_shapes

:@`*
dtype0�
sequential_20/gru_7/IdentityIdentity/sequential_20/gru_7/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:@`�
)sequential_20/gru_7/Read_1/ReadVariableOpReadVariableOp2sequential_20_gru_7_read_1_readvariableop_resource*
_output_shapes

: `*
dtype0�
sequential_20/gru_7/Identity_1Identity1sequential_20/gru_7/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `�
)sequential_20/gru_7/Read_2/ReadVariableOpReadVariableOp2sequential_20_gru_7_read_2_readvariableop_resource*
_output_shapes

:`*
dtype0�
sequential_20/gru_7/Identity_2Identity1sequential_20/gru_7/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`�
#sequential_20/gru_7/PartitionedCallPartitionedCall*sequential_20/simple_rnn_6/transpose_1:y:0"sequential_20/gru_7/zeros:output:0%sequential_20/gru_7/Identity:output:0'sequential_20/gru_7/Identity_1:output:0'sequential_20/gru_7/Identity_2:output:0*
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
 __inference_standard_gru_2720835�
,sequential_20/dense_28/MatMul/ReadVariableOpReadVariableOp5sequential_20_dense_28_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
sequential_20/dense_28/MatMulMatMul,sequential_20/gru_7/PartitionedCall:output:04sequential_20/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-sequential_20/dense_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_20/dense_28/BiasAddBiasAdd'sequential_20/dense_28/MatMul:product:05sequential_20/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,sequential_20/dense_29/MatMul/ReadVariableOpReadVariableOp5sequential_20_dense_29_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential_20/dense_29/MatMulMatMul'sequential_20/dense_28/BiasAdd:output:04sequential_20/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-sequential_20/dense_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_20/dense_29/BiasAddBiasAdd'sequential_20/dense_29/MatMul:product:05sequential_20/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
IdentityIdentity'sequential_20/dense_29/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^sequential_20/conv1d_8/BiasAdd/ReadVariableOp:^sequential_20/conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp.^sequential_20/dense_28/BiasAdd/ReadVariableOp-^sequential_20/dense_28/MatMul/ReadVariableOp.^sequential_20/dense_29/BiasAdd/ReadVariableOp-^sequential_20/dense_29/MatMul/ReadVariableOp(^sequential_20/gru_7/Read/ReadVariableOp*^sequential_20/gru_7/Read_1/ReadVariableOp*^sequential_20/gru_7/Read_2/ReadVariableOpE^sequential_20/simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOpD^sequential_20/simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOpF^sequential_20/simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOp!^sequential_20/simple_rnn_6/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2^
-sequential_20/conv1d_8/BiasAdd/ReadVariableOp-sequential_20/conv1d_8/BiasAdd/ReadVariableOp2v
9sequential_20/conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp9sequential_20/conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp2^
-sequential_20/dense_28/BiasAdd/ReadVariableOp-sequential_20/dense_28/BiasAdd/ReadVariableOp2\
,sequential_20/dense_28/MatMul/ReadVariableOp,sequential_20/dense_28/MatMul/ReadVariableOp2^
-sequential_20/dense_29/BiasAdd/ReadVariableOp-sequential_20/dense_29/BiasAdd/ReadVariableOp2\
,sequential_20/dense_29/MatMul/ReadVariableOp,sequential_20/dense_29/MatMul/ReadVariableOp2R
'sequential_20/gru_7/Read/ReadVariableOp'sequential_20/gru_7/Read/ReadVariableOp2V
)sequential_20/gru_7/Read_1/ReadVariableOp)sequential_20/gru_7/Read_1/ReadVariableOp2V
)sequential_20/gru_7/Read_2/ReadVariableOp)sequential_20/gru_7/Read_2/ReadVariableOp2�
Dsequential_20/simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOpDsequential_20/simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOp2�
Csequential_20/simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOpCsequential_20/simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOp2�
Esequential_20/simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOpEsequential_20/simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOp2D
 sequential_20/simple_rnn_6/while sequential_20/simple_rnn_6/while:[ W
+
_output_shapes
:���������
(
_user_specified_nameconv1d_8_input
�>
�
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2724825
inputs_0C
1simple_rnn_cell_12_matmul_readvariableop_resource: @@
2simple_rnn_cell_12_biasadd_readvariableop_resource:@E
3simple_rnn_cell_12_matmul_1_readvariableop_resource:@@
identity��)simple_rnn_cell_12/BiasAdd/ReadVariableOp�(simple_rnn_cell_12/MatMul/ReadVariableOp�*simple_rnn_cell_12/MatMul_1/ReadVariableOp�while=
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
 :������������������ D
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
valueB"����    �
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
:��������� *
shrink_axis_mask�
(simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_12_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0�
simple_rnn_cell_12/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell_12/BiasAddBiasAdd#simple_rnn_cell_12/MatMul:product:01simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_12_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell_12/MatMul_1MatMulzeros:output:02simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell_12/addAddV2#simple_rnn_cell_12/BiasAdd:output:0%simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@m
simple_rnn_cell_12/TanhTanhsimple_rnn_cell_12/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_12_matmul_readvariableop_resource2simple_rnn_cell_12_biasadd_readvariableop_resource3simple_rnn_cell_12_matmul_1_readvariableop_resource*
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
while_body_2724759*
condR
while_cond_2724758*8
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
NoOpNoOp*^simple_rnn_cell_12/BiasAdd/ReadVariableOp)^simple_rnn_cell_12/MatMul/ReadVariableOp+^simple_rnn_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������ : : : 2V
)simple_rnn_cell_12/BiasAdd/ReadVariableOp)simple_rnn_cell_12/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_12/MatMul/ReadVariableOp(simple_rnn_cell_12/MatMul/ReadVariableOp2X
*simple_rnn_cell_12/MatMul_1/ReadVariableOp*simple_rnn_cell_12/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������ 
"
_user_specified_name
inputs/0
�,
�
while_body_2724867
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_12_matmul_readvariableop_resource_0: @H
:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_12_matmul_readvariableop_resource: @F
8while_simple_rnn_cell_12_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_12_matmul_1_readvariableop_resource:@@��/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp�.while/simple_rnn_cell_12/MatMul/ReadVariableOp�0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype0�
.while/simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_12_matmul_readvariableop_resource_0*
_output_shapes

: @*
dtype0�
while/simple_rnn_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/while/simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
 while/simple_rnn_cell_12/BiasAddBiasAdd)while/simple_rnn_cell_12/MatMul:product:07while/simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0while/simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
!while/simple_rnn_cell_12/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell_12/addAddV2)while/simple_rnn_cell_12/BiasAdd:output:0+while/simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@y
while/simple_rnn_cell_12/TanhTanh while/simple_rnn_cell_12/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_12/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_12/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp0^while/simple_rnn_cell_12/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_12/MatMul/ReadVariableOp1^while/simple_rnn_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_12_biasadd_readvariableop_resource:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_12_matmul_1_readvariableop_resource;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_12_matmul_readvariableop_resource9while_simple_rnn_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2b
/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_12/MatMul/ReadVariableOp.while/simple_rnn_cell_12/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp: 
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
'__forward_gpu_gru_with_fallback_2725568

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
api_implements*(gru_512f6324-f693-4328-96e4-47ae443c807d*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2725433_2725569*
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
�
�
B__inference_gru_7_layer_call_and_return_conditional_losses_2725571
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
 __inference_standard_gru_2725356i

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
�,
�
while_body_2725083
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_12_matmul_readvariableop_resource_0: @H
:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_12_matmul_readvariableop_resource: @F
8while_simple_rnn_cell_12_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_12_matmul_1_readvariableop_resource:@@��/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp�.while/simple_rnn_cell_12/MatMul/ReadVariableOp�0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype0�
.while/simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_12_matmul_readvariableop_resource_0*
_output_shapes

: @*
dtype0�
while/simple_rnn_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/while/simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
 while/simple_rnn_cell_12/BiasAddBiasAdd)while/simple_rnn_cell_12/MatMul:product:07while/simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0while/simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
!while/simple_rnn_cell_12/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell_12/addAddV2)while/simple_rnn_cell_12/BiasAdd:output:0+while/simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@y
while/simple_rnn_cell_12/TanhTanh while/simple_rnn_cell_12/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_12/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_12/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp0^while/simple_rnn_cell_12/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_12/MatMul/ReadVariableOp1^while/simple_rnn_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_12_biasadd_readvariableop_resource:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_12_matmul_1_readvariableop_resource;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_12_matmul_readvariableop_resource9while_simple_rnn_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2b
/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_12/MatMul/ReadVariableOp.while/simple_rnn_cell_12/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp: 
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
��
�

<__inference___backward_gpu_gru_with_fallback_2720912_2721048
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
:��������� *
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
7:���������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������@u
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
:���������@t

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
�:��������� :��������� :��������� : :��������� :��������� :��������� : ::���������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_cb1cd7d5-d1d7-4891-9a30-91ba2b88398b*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2721047*
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
:��������� :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:���������@:1
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
'__forward_gpu_gru_with_fallback_2722137

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
api_implements*(gru_e1fd2348-df97-440c-a602-69c0eee8e6d8*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2722002_2722138*
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
�
E__inference_dense_29_layer_call_and_return_conditional_losses_2722713

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
while_body_2722228
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_12_matmul_readvariableop_resource_0: @H
:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_12_matmul_readvariableop_resource: @F
8while_simple_rnn_cell_12_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_12_matmul_1_readvariableop_resource:@@��/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp�.while/simple_rnn_cell_12/MatMul/ReadVariableOp�0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype0�
.while/simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_12_matmul_readvariableop_resource_0*
_output_shapes

: @*
dtype0�
while/simple_rnn_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/while/simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
 while/simple_rnn_cell_12/BiasAddBiasAdd)while/simple_rnn_cell_12/MatMul:product:07while/simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0while/simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
!while/simple_rnn_cell_12/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell_12/addAddV2)while/simple_rnn_cell_12/BiasAdd:output:0+while/simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@y
while/simple_rnn_cell_12/TanhTanh while/simple_rnn_cell_12/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_12/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_12/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp0^while/simple_rnn_cell_12/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_12/MatMul/ReadVariableOp1^while/simple_rnn_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_12_biasadd_readvariableop_resource:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_12_matmul_1_readvariableop_resource;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_12_matmul_readvariableop_resource9while_simple_rnn_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2b
/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_12/MatMul/ReadVariableOp.while/simple_rnn_cell_12/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp: 
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
while_body_2725644
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
 __inference_standard_gru_2722943

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
:���������@B
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
while_body_2722853*
condR
while_cond_2722852*R
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_b5d7929a-c103-4256-9d6a-9fca617f4787*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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

<__inference___backward_gpu_gru_with_fallback_2722541_2722677
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
:��������� *
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
7:���������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������@u
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
:���������@t

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
�:��������� :��������� :��������� : :��������� :��������� :��������� : ::���������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_71467f37-2766-43bf-ab1c-686fb15836fa*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2722676*
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
:��������� :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:���������@:1
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
�
�
B__inference_gru_7_layer_call_and_return_conditional_losses_2722679

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
 __inference_standard_gru_2722464i

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
:���������@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
Ɠ
�

<__inference___backward_gpu_gru_with_fallback_2721613_2721749
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
api_implements*(gru_298461b9-b437-48ed-afb0-0a379bc27613*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2721748*
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
�
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2724933
inputs_0C
1simple_rnn_cell_12_matmul_readvariableop_resource: @@
2simple_rnn_cell_12_biasadd_readvariableop_resource:@E
3simple_rnn_cell_12_matmul_1_readvariableop_resource:@@
identity��)simple_rnn_cell_12/BiasAdd/ReadVariableOp�(simple_rnn_cell_12/MatMul/ReadVariableOp�*simple_rnn_cell_12/MatMul_1/ReadVariableOp�while=
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
 :������������������ D
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
valueB"����    �
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
:��������� *
shrink_axis_mask�
(simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_12_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0�
simple_rnn_cell_12/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell_12/BiasAddBiasAdd#simple_rnn_cell_12/MatMul:product:01simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_12_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell_12/MatMul_1MatMulzeros:output:02simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell_12/addAddV2#simple_rnn_cell_12/BiasAdd:output:0%simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@m
simple_rnn_cell_12/TanhTanhsimple_rnn_cell_12/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_12_matmul_readvariableop_resource2simple_rnn_cell_12_biasadd_readvariableop_resource3simple_rnn_cell_12_matmul_1_readvariableop_resource*
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
while_body_2724867*
condR
while_cond_2724866*8
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
NoOpNoOp*^simple_rnn_cell_12/BiasAdd/ReadVariableOp)^simple_rnn_cell_12/MatMul/ReadVariableOp+^simple_rnn_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������ : : : 2V
)simple_rnn_cell_12/BiasAdd/ReadVariableOp)simple_rnn_cell_12/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_12/MatMul/ReadVariableOp(simple_rnn_cell_12/MatMul/ReadVariableOp2X
*simple_rnn_cell_12/MatMul_1/ReadVariableOp*simple_rnn_cell_12/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������ 
"
_user_specified_name
inputs/0
�-
�
while_body_2725266
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
�
'__forward_gpu_gru_with_fallback_2725946

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
api_implements*(gru_0ad6a64d-677a-42fe-949d-ced9d9b57ca8*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2725811_2725947*
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
�,
�
while_body_2723222
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_12_matmul_readvariableop_resource_0: @H
:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_12_matmul_readvariableop_resource: @F
8while_simple_rnn_cell_12_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_12_matmul_1_readvariableop_resource:@@��/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp�.while/simple_rnn_cell_12/MatMul/ReadVariableOp�0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype0�
.while/simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_12_matmul_readvariableop_resource_0*
_output_shapes

: @*
dtype0�
while/simple_rnn_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/while/simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
 while/simple_rnn_cell_12/BiasAddBiasAdd)while/simple_rnn_cell_12/MatMul:product:07while/simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0while/simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
!while/simple_rnn_cell_12/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell_12/addAddV2)while/simple_rnn_cell_12/BiasAdd:output:0+while/simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@y
while/simple_rnn_cell_12/TanhTanh while/simple_rnn_cell_12/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_12/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_12/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp0^while/simple_rnn_cell_12/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_12/MatMul/ReadVariableOp1^while/simple_rnn_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_12_biasadd_readvariableop_resource:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_12_matmul_1_readvariableop_resource;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_12_matmul_readvariableop_resource9while_simple_rnn_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2b
/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_12/MatMul/ReadVariableOp.while/simple_rnn_cell_12/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp: 
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
�
�
*__inference_conv1d_8_layer_call_fn_2724633

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv1d_8_layer_call_and_return_conditional_losses_2722172s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
��
�

<__inference___backward_gpu_gru_with_fallback_2723962_2724098
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
:��������� *
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
7:���������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������@u
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
:���������@t

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
�:��������� :��������� :��������� : :��������� :��������� :��������� : ::���������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_f7d4556a-b6c2-4fff-93ae-297efbd138b2*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2724097*
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
:��������� :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:���������@:1
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

�
simple_rnn_6_while_cond_27241716
2simple_rnn_6_while_simple_rnn_6_while_loop_counter<
8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations"
simple_rnn_6_while_placeholder$
 simple_rnn_6_while_placeholder_1$
 simple_rnn_6_while_placeholder_28
4simple_rnn_6_while_less_simple_rnn_6_strided_slice_1O
Ksimple_rnn_6_while_simple_rnn_6_while_cond_2724171___redundant_placeholder0O
Ksimple_rnn_6_while_simple_rnn_6_while_cond_2724171___redundant_placeholder1O
Ksimple_rnn_6_while_simple_rnn_6_while_cond_2724171___redundant_placeholder2O
Ksimple_rnn_6_while_simple_rnn_6_while_cond_2724171___redundant_placeholder3
simple_rnn_6_while_identity
�
simple_rnn_6/while/LessLesssimple_rnn_6_while_placeholder4simple_rnn_6_while_less_simple_rnn_6_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_6/while/IdentityIdentitysimple_rnn_6/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_6_while_identity$simple_rnn_6/while/Identity:output:0*(
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
E__inference_dense_28_layer_call_and_return_conditional_losses_2726724

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
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
�
�
while_cond_2723221
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2723221___redundant_placeholder05
1while_while_cond_2723221___redundant_placeholder15
1while_while_cond_2723221___redundant_placeholder25
1while_while_cond_2723221___redundant_placeholder3
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
�-
�
while_body_2726400
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
�-
�
while_body_2723795
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
while_cond_2726021
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2726021___redundant_placeholder05
1while_while_cond_2726021___redundant_placeholder15
1while_while_cond_2726021___redundant_placeholder25
1while_while_cond_2726021___redundant_placeholder35
1while_while_cond_2726021___redundant_placeholder4
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

�
4__inference_simple_rnn_cell_12_layer_call_fn_2726771

inputs
states_0
unknown: @
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
O__inference_simple_rnn_cell_12_layer_call_and_return_conditional_losses_2721245o
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
,:��������� :���������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0
�>
�
 __inference_standard_gru_2725734

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
while_body_2725644*
condR
while_cond_2725643*R
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
api_implements*(gru_0ad6a64d-677a-42fe-949d-ced9d9b57ca8*
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
while_cond_2724866
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2724866___redundant_placeholder05
1while_while_cond_2724866___redundant_placeholder15
1while_while_cond_2724866___redundant_placeholder25
1while_while_cond_2724866___redundant_placeholder3
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
 __inference_standard_gru_2723885

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
:���������@B
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
while_body_2723795*
condR
while_cond_2723794*R
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_f7d4556a-b6c2-4fff-93ae-297efbd138b2*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
while_cond_2725082
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2725082___redundant_placeholder05
1while_while_cond_2725082___redundant_placeholder15
1while_while_cond_2725082___redundant_placeholder25
1while_while_cond_2725082___redundant_placeholder3
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
'__forward_gpu_gru_with_fallback_2724097

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
4:��������� :��������� : :*
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_f7d4556a-b6c2-4fff-93ae-297efbd138b2*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2723962_2724098*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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

<__inference___backward_gpu_gru_with_fallback_2723020_2723156
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
:��������� *
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
7:���������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������@u
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
:���������@t

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
�:��������� :��������� :��������� : :��������� :��������� :��������� : ::���������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_b5d7929a-c103-4256-9d6a-9fca617f4787*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2723155*
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
:��������� :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:���������@:1
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
�
�
B__inference_gru_7_layer_call_and_return_conditional_losses_2723158

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
 __inference_standard_gru_2722943i

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
:���������@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
while_cond_2722852
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2722852___redundant_placeholder05
1while_while_cond_2722852___redundant_placeholder15
1while_while_cond_2722852___redundant_placeholder25
1while_while_cond_2722852___redundant_placeholder35
1while_while_cond_2722852___redundant_placeholder4
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
�
�
.__inference_simple_rnn_6_layer_call_fn_2724717

inputs
unknown: @
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
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2723288s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:��������� : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
.__inference_simple_rnn_6_layer_call_fn_2724695
inputs_0
unknown: @
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
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2721360|
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
&:������������������ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������ 
"
_user_specified_name
inputs/0
�E
�
-sequential_20_simple_rnn_6_while_body_2720610R
Nsequential_20_simple_rnn_6_while_sequential_20_simple_rnn_6_while_loop_counterX
Tsequential_20_simple_rnn_6_while_sequential_20_simple_rnn_6_while_maximum_iterations0
,sequential_20_simple_rnn_6_while_placeholder2
.sequential_20_simple_rnn_6_while_placeholder_12
.sequential_20_simple_rnn_6_while_placeholder_2Q
Msequential_20_simple_rnn_6_while_sequential_20_simple_rnn_6_strided_slice_1_0�
�sequential_20_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_20_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0f
Tsequential_20_simple_rnn_6_while_simple_rnn_cell_12_matmul_readvariableop_resource_0: @c
Usequential_20_simple_rnn_6_while_simple_rnn_cell_12_biasadd_readvariableop_resource_0:@h
Vsequential_20_simple_rnn_6_while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0:@@-
)sequential_20_simple_rnn_6_while_identity/
+sequential_20_simple_rnn_6_while_identity_1/
+sequential_20_simple_rnn_6_while_identity_2/
+sequential_20_simple_rnn_6_while_identity_3/
+sequential_20_simple_rnn_6_while_identity_4O
Ksequential_20_simple_rnn_6_while_sequential_20_simple_rnn_6_strided_slice_1�
�sequential_20_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_20_simple_rnn_6_tensorarrayunstack_tensorlistfromtensord
Rsequential_20_simple_rnn_6_while_simple_rnn_cell_12_matmul_readvariableop_resource: @a
Ssequential_20_simple_rnn_6_while_simple_rnn_cell_12_biasadd_readvariableop_resource:@f
Tsequential_20_simple_rnn_6_while_simple_rnn_cell_12_matmul_1_readvariableop_resource:@@��Jsequential_20/simple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp�Isequential_20/simple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOp�Ksequential_20/simple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOp�
Rsequential_20/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
Dsequential_20/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_20_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_20_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0,sequential_20_simple_rnn_6_while_placeholder[sequential_20/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype0�
Isequential_20/simple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOpTsequential_20_simple_rnn_6_while_simple_rnn_cell_12_matmul_readvariableop_resource_0*
_output_shapes

: @*
dtype0�
:sequential_20/simple_rnn_6/while/simple_rnn_cell_12/MatMulMatMulKsequential_20/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem:item:0Qsequential_20/simple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Jsequential_20/simple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOpUsequential_20_simple_rnn_6_while_simple_rnn_cell_12_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
;sequential_20/simple_rnn_6/while/simple_rnn_cell_12/BiasAddBiasAddDsequential_20/simple_rnn_6/while/simple_rnn_cell_12/MatMul:product:0Rsequential_20/simple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Ksequential_20/simple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOpVsequential_20_simple_rnn_6_while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
<sequential_20/simple_rnn_6/while/simple_rnn_cell_12/MatMul_1MatMul.sequential_20_simple_rnn_6_while_placeholder_2Ssequential_20/simple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
7sequential_20/simple_rnn_6/while/simple_rnn_cell_12/addAddV2Dsequential_20/simple_rnn_6/while/simple_rnn_cell_12/BiasAdd:output:0Fsequential_20/simple_rnn_6/while/simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
8sequential_20/simple_rnn_6/while/simple_rnn_cell_12/TanhTanh;sequential_20/simple_rnn_6/while/simple_rnn_cell_12/add:z:0*
T0*'
_output_shapes
:���������@�
Esequential_20/simple_rnn_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_20_simple_rnn_6_while_placeholder_1,sequential_20_simple_rnn_6_while_placeholder<sequential_20/simple_rnn_6/while/simple_rnn_cell_12/Tanh:y:0*
_output_shapes
: *
element_dtype0:���h
&sequential_20/simple_rnn_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
$sequential_20/simple_rnn_6/while/addAddV2,sequential_20_simple_rnn_6_while_placeholder/sequential_20/simple_rnn_6/while/add/y:output:0*
T0*
_output_shapes
: j
(sequential_20/simple_rnn_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
&sequential_20/simple_rnn_6/while/add_1AddV2Nsequential_20_simple_rnn_6_while_sequential_20_simple_rnn_6_while_loop_counter1sequential_20/simple_rnn_6/while/add_1/y:output:0*
T0*
_output_shapes
: �
)sequential_20/simple_rnn_6/while/IdentityIdentity*sequential_20/simple_rnn_6/while/add_1:z:0&^sequential_20/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: �
+sequential_20/simple_rnn_6/while/Identity_1IdentityTsequential_20_simple_rnn_6_while_sequential_20_simple_rnn_6_while_maximum_iterations&^sequential_20/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: �
+sequential_20/simple_rnn_6/while/Identity_2Identity(sequential_20/simple_rnn_6/while/add:z:0&^sequential_20/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: �
+sequential_20/simple_rnn_6/while/Identity_3IdentityUsequential_20/simple_rnn_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^sequential_20/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: �
+sequential_20/simple_rnn_6/while/Identity_4Identity<sequential_20/simple_rnn_6/while/simple_rnn_cell_12/Tanh:y:0&^sequential_20/simple_rnn_6/while/NoOp*
T0*'
_output_shapes
:���������@�
%sequential_20/simple_rnn_6/while/NoOpNoOpK^sequential_20/simple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOpJ^sequential_20/simple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOpL^sequential_20/simple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "_
)sequential_20_simple_rnn_6_while_identity2sequential_20/simple_rnn_6/while/Identity:output:0"c
+sequential_20_simple_rnn_6_while_identity_14sequential_20/simple_rnn_6/while/Identity_1:output:0"c
+sequential_20_simple_rnn_6_while_identity_24sequential_20/simple_rnn_6/while/Identity_2:output:0"c
+sequential_20_simple_rnn_6_while_identity_34sequential_20/simple_rnn_6/while/Identity_3:output:0"c
+sequential_20_simple_rnn_6_while_identity_44sequential_20/simple_rnn_6/while/Identity_4:output:0"�
Ksequential_20_simple_rnn_6_while_sequential_20_simple_rnn_6_strided_slice_1Msequential_20_simple_rnn_6_while_sequential_20_simple_rnn_6_strided_slice_1_0"�
Ssequential_20_simple_rnn_6_while_simple_rnn_cell_12_biasadd_readvariableop_resourceUsequential_20_simple_rnn_6_while_simple_rnn_cell_12_biasadd_readvariableop_resource_0"�
Tsequential_20_simple_rnn_6_while_simple_rnn_cell_12_matmul_1_readvariableop_resourceVsequential_20_simple_rnn_6_while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0"�
Rsequential_20_simple_rnn_6_while_simple_rnn_cell_12_matmul_readvariableop_resourceTsequential_20_simple_rnn_6_while_simple_rnn_cell_12_matmul_readvariableop_resource_0"�
�sequential_20_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_20_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor�sequential_20_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_20_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2�
Jsequential_20/simple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOpJsequential_20/simple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp2�
Isequential_20/simple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOpIsequential_20/simple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOp2�
Ksequential_20/simple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOpKsequential_20/simple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOp: 
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
�
�
while_cond_2721137
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2721137___redundant_placeholder05
1while_while_cond_2721137___redundant_placeholder15
1while_while_cond_2721137___redundant_placeholder25
1while_while_cond_2721137___redundant_placeholder3
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
�8
�
simple_rnn_6_while_body_27241726
2simple_rnn_6_while_simple_rnn_6_while_loop_counter<
8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations"
simple_rnn_6_while_placeholder$
 simple_rnn_6_while_placeholder_1$
 simple_rnn_6_while_placeholder_25
1simple_rnn_6_while_simple_rnn_6_strided_slice_1_0q
msimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0X
Fsimple_rnn_6_while_simple_rnn_cell_12_matmul_readvariableop_resource_0: @U
Gsimple_rnn_6_while_simple_rnn_cell_12_biasadd_readvariableop_resource_0:@Z
Hsimple_rnn_6_while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0:@@
simple_rnn_6_while_identity!
simple_rnn_6_while_identity_1!
simple_rnn_6_while_identity_2!
simple_rnn_6_while_identity_3!
simple_rnn_6_while_identity_43
/simple_rnn_6_while_simple_rnn_6_strided_slice_1o
ksimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensorV
Dsimple_rnn_6_while_simple_rnn_cell_12_matmul_readvariableop_resource: @S
Esimple_rnn_6_while_simple_rnn_cell_12_biasadd_readvariableop_resource:@X
Fsimple_rnn_6_while_simple_rnn_cell_12_matmul_1_readvariableop_resource:@@��<simple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp�;simple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOp�=simple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOp�
Dsimple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
6simple_rnn_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_6_while_placeholderMsimple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype0�
;simple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOpFsimple_rnn_6_while_simple_rnn_cell_12_matmul_readvariableop_resource_0*
_output_shapes

: @*
dtype0�
,simple_rnn_6/while/simple_rnn_cell_12/MatMulMatMul=simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem:item:0Csimple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<simple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOpGsimple_rnn_6_while_simple_rnn_cell_12_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
-simple_rnn_6/while/simple_rnn_cell_12/BiasAddBiasAdd6simple_rnn_6/while/simple_rnn_cell_12/MatMul:product:0Dsimple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
=simple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOpHsimple_rnn_6_while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
.simple_rnn_6/while/simple_rnn_cell_12/MatMul_1MatMul simple_rnn_6_while_placeholder_2Esimple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)simple_rnn_6/while/simple_rnn_cell_12/addAddV26simple_rnn_6/while/simple_rnn_cell_12/BiasAdd:output:08simple_rnn_6/while/simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
*simple_rnn_6/while/simple_rnn_cell_12/TanhTanh-simple_rnn_6/while/simple_rnn_cell_12/add:z:0*
T0*'
_output_shapes
:���������@�
7simple_rnn_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_6_while_placeholder_1simple_rnn_6_while_placeholder.simple_rnn_6/while/simple_rnn_cell_12/Tanh:y:0*
_output_shapes
: *
element_dtype0:���Z
simple_rnn_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_6/while/addAddV2simple_rnn_6_while_placeholder!simple_rnn_6/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_6/while/add_1AddV22simple_rnn_6_while_simple_rnn_6_while_loop_counter#simple_rnn_6/while/add_1/y:output:0*
T0*
_output_shapes
: �
simple_rnn_6/while/IdentityIdentitysimple_rnn_6/while/add_1:z:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_6/while/Identity_1Identity8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_6/while/Identity_2Identitysimple_rnn_6/while/add:z:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_6/while/Identity_3IdentityGsimple_rnn_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_6/while/Identity_4Identity.simple_rnn_6/while/simple_rnn_cell_12/Tanh:y:0^simple_rnn_6/while/NoOp*
T0*'
_output_shapes
:���������@�
simple_rnn_6/while/NoOpNoOp=^simple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp<^simple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOp>^simple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_6_while_identity$simple_rnn_6/while/Identity:output:0"G
simple_rnn_6_while_identity_1&simple_rnn_6/while/Identity_1:output:0"G
simple_rnn_6_while_identity_2&simple_rnn_6/while/Identity_2:output:0"G
simple_rnn_6_while_identity_3&simple_rnn_6/while/Identity_3:output:0"G
simple_rnn_6_while_identity_4&simple_rnn_6/while/Identity_4:output:0"d
/simple_rnn_6_while_simple_rnn_6_strided_slice_11simple_rnn_6_while_simple_rnn_6_strided_slice_1_0"�
Esimple_rnn_6_while_simple_rnn_cell_12_biasadd_readvariableop_resourceGsimple_rnn_6_while_simple_rnn_cell_12_biasadd_readvariableop_resource_0"�
Fsimple_rnn_6_while_simple_rnn_cell_12_matmul_1_readvariableop_resourceHsimple_rnn_6_while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0"�
Dsimple_rnn_6_while_simple_rnn_cell_12_matmul_readvariableop_resourceFsimple_rnn_6_while_simple_rnn_cell_12_matmul_readvariableop_resource_0"�
ksimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensormsimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2|
<simple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp<simple_rnn_6/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp2z
;simple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOp;simple_rnn_6/while/simple_rnn_cell_12/MatMul/ReadVariableOp2~
=simple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOp=simple_rnn_6/while/simple_rnn_cell_12/MatMul_1/ReadVariableOp: 
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
�
while_cond_2721445
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2721445___redundant_placeholder05
1while_while_cond_2721445___redundant_placeholder15
1while_while_cond_2721445___redundant_placeholder25
1while_while_cond_2721445___redundant_placeholder35
1while_while_cond_2721445___redundant_placeholder4
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
�
'__forward_gpu_gru_with_fallback_2721748

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
api_implements*(gru_298461b9-b437-48ed-afb0-0a379bc27613*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2721613_2721749*
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
)__inference_gpu_gru_with_fallback_2726566

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
:���������@P
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
4:��������� :��������� : :*
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_72876175-a1f8-456c-a1ea-e2cc127d7930*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
O__inference_simple_rnn_cell_12_layer_call_and_return_conditional_losses_2726805

inputs
states_00
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
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
,:��������� :���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0
�-
�
while_body_2722374
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
�
'__forward_gpu_gru_with_fallback_2726324

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
4:��������� :��������� : :*
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_4c975bea-1b37-48d1-ab33-42b4dd45878b*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2726189_2726325*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
while_body_2726022
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
while_cond_2725643
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2725643___redundant_placeholder05
1while_while_cond_2725643___redundant_placeholder15
1while_while_cond_2725643___redundant_placeholder25
1while_while_cond_2725643___redundant_placeholder35
1while_while_cond_2725643___redundant_placeholder4
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
)__inference_gpu_gru_with_fallback_2722540

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
:���������@P
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
4:��������� :��������� : :*
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_71467f37-2766-43bf-ab1c-686fb15836fa*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
)__inference_gpu_gru_with_fallback_2721612

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
api_implements*(gru_298461b9-b437-48ed-afb0-0a379bc27613*
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
�
)__inference_gpu_gru_with_fallback_2720911

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
:���������@P
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
4:��������� :��������� : :*
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_cb1cd7d5-d1d7-4891-9a30-91ba2b88398b*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
�
E__inference_dense_29_layer_call_and_return_conditional_losses_2726743

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_29_layer_call_fn_2726733

inputs
unknown:
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
E__inference_dense_29_layer_call_and_return_conditional_losses_2722713o
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
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_simple_rnn_6_layer_call_fn_2724706

inputs
unknown: @
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
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2722294s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:��������� : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
while_cond_2724974
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_2724974___redundant_placeholder05
1while_while_cond_2724974___redundant_placeholder15
1while_while_cond_2724974___redundant_placeholder25
1while_while_cond_2724974___redundant_placeholder3
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
�
while_cond_2720744
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2720744___redundant_placeholder05
1while_while_cond_2720744___redundant_placeholder15
1while_while_cond_2720744___redundant_placeholder25
1while_while_cond_2720744___redundant_placeholder35
1while_while_cond_2720744___redundant_placeholder4
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
�
�
*__inference_dense_28_layer_call_fn_2726714

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_2722697o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
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
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2725041

inputsC
1simple_rnn_cell_12_matmul_readvariableop_resource: @@
2simple_rnn_cell_12_biasadd_readvariableop_resource:@E
3simple_rnn_cell_12_matmul_1_readvariableop_resource:@@
identity��)simple_rnn_cell_12/BiasAdd/ReadVariableOp�(simple_rnn_cell_12/MatMul/ReadVariableOp�*simple_rnn_cell_12/MatMul_1/ReadVariableOp�while;
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
:��������� D
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
valueB"����    �
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
:��������� *
shrink_axis_mask�
(simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_12_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0�
simple_rnn_cell_12/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell_12/BiasAddBiasAdd#simple_rnn_cell_12/MatMul:product:01simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_12_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell_12/MatMul_1MatMulzeros:output:02simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell_12/addAddV2#simple_rnn_cell_12/BiasAdd:output:0%simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@m
simple_rnn_cell_12/TanhTanhsimple_rnn_cell_12/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_12_matmul_readvariableop_resource2simple_rnn_cell_12_biasadd_readvariableop_resource3simple_rnn_cell_12_matmul_1_readvariableop_resource*
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
while_body_2724975*
condR
while_cond_2724974*8
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
:���������@*
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
:���������@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp*^simple_rnn_cell_12/BiasAdd/ReadVariableOp)^simple_rnn_cell_12/MatMul/ReadVariableOp+^simple_rnn_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:��������� : : : 2V
)simple_rnn_cell_12/BiasAdd/ReadVariableOp)simple_rnn_cell_12/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_12/MatMul/ReadVariableOp(simple_rnn_cell_12/MatMul/ReadVariableOp2X
*simple_rnn_cell_12/MatMul_1/ReadVariableOp*simple_rnn_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�>
�
'__forward_gpu_gru_with_fallback_2721047

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
4:��������� :��������� : :*
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_cb1cd7d5-d1d7-4891-9a30-91ba2b88398b*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2720912_2721048*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
�
 __inference_standard_gru_2722464

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
:���������@B
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
while_body_2722374*
condR
while_cond_2722373*R
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_71467f37-2766-43bf-ab1c-686fb15836fa*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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

<__inference___backward_gpu_gru_with_fallback_2726189_2726325
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
:��������� *
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
7:���������@:��������� : :�I*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������@u
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
:���������@t

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
�:��������� :��������� :��������� : :��������� :��������� :��������� : ::���������@:��������� : :�I::��������� : ::::::: : : *<
api_implements*(gru_4c975bea-1b37-48d1-ab33-42b4dd45878b*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2726324*
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
:��������� :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:���������@:1
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
�
 __inference_standard_gru_2726490

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
:���������@B
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
while_body_2726400*
condR
while_cond_2726399*R
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_72876175-a1f8-456c-a1ea-e2cc127d7930*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
)__inference_gpu_gru_with_fallback_2725810

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
api_implements*(gru_0ad6a64d-677a-42fe-949d-ced9d9b57ca8*
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
�!
�
while_body_2721297
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_12_2721319_0: @0
"while_simple_rnn_cell_12_2721321_0:@4
"while_simple_rnn_cell_12_2721323_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_12_2721319: @.
 while_simple_rnn_cell_12_2721321:@2
 while_simple_rnn_cell_12_2721323:@@��0while/simple_rnn_cell_12/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype0�
0while/simple_rnn_cell_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_12_2721319_0"while_simple_rnn_cell_12_2721321_0"while_simple_rnn_cell_12_2721323_0*
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
O__inference_simple_rnn_cell_12_layer_call_and_return_conditional_losses_2721245�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_12/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity9while/simple_rnn_cell_12/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@

while/NoOpNoOp1^while/simple_rnn_cell_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_12_2721319"while_simple_rnn_cell_12_2721319_0"F
 while_simple_rnn_cell_12_2721321"while_simple_rnn_cell_12_2721321_0"F
 while_simple_rnn_cell_12_2721323"while_simple_rnn_cell_12_2721323_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2d
0while/simple_rnn_cell_12/StatefulPartitionedCall0while/simple_rnn_cell_12/StatefulPartitionedCall: 
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
.__inference_simple_rnn_6_layer_call_fn_2724684
inputs_0
unknown: @
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
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2721201|
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
&:������������������ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������ 
"
_user_specified_name
inputs/0
�>
�
'__forward_gpu_gru_with_fallback_2722676

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
4:��������� :��������� : :*
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_71467f37-2766-43bf-ab1c-686fb15836fa*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2722541_2722677*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
Ɠ
�

<__inference___backward_gpu_gru_with_fallback_2725811_2725947
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
api_implements*(gru_0ad6a64d-677a-42fe-949d-ced9d9b57ca8*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2725946*
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
�
 __inference_standard_gru_2726112

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
:���������@B
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
while_body_2726022*
condR
while_cond_2726021*R
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_4c975bea-1b37-48d1-ab33-42b4dd45878b*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
�"
�
J__inference_sequential_20_layer_call_and_return_conditional_losses_2722720

inputs&
conv1d_8_2722173: 
conv1d_8_2722175: &
simple_rnn_6_2722295: @"
simple_rnn_6_2722297:@&
simple_rnn_6_2722299:@@
gru_7_2722680:@`
gru_7_2722682: `
gru_7_2722684:`"
dense_28_2722698: 
dense_28_2722700:"
dense_29_2722714:
dense_29_2722716:
identity�� conv1d_8/StatefulPartitionedCall� dense_28/StatefulPartitionedCall� dense_29/StatefulPartitionedCall�gru_7/StatefulPartitionedCall�$simple_rnn_6/StatefulPartitionedCall�
 conv1d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_8_2722173conv1d_8_2722175*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv1d_8_layer_call_and_return_conditional_losses_2722172�
flatten_9/PartitionedCallPartitionedCall)conv1d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_2722184�
repeat_vector_2/PartitionedCallPartitionedCall"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_2721074�
$simple_rnn_6/StatefulPartitionedCallStatefulPartitionedCall(repeat_vector_2/PartitionedCall:output:0simple_rnn_6_2722295simple_rnn_6_2722297simple_rnn_6_2722299*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2722294�
gru_7/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_6/StatefulPartitionedCall:output:0gru_7_2722680gru_7_2722682gru_7_2722684*
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
B__inference_gru_7_layer_call_and_return_conditional_losses_2722679�
 dense_28/StatefulPartitionedCallStatefulPartitionedCall&gru_7/StatefulPartitionedCall:output:0dense_28_2722698dense_28_2722700*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_2722697�
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_2722714dense_29_2722716*
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
E__inference_dense_29_layer_call_and_return_conditional_losses_2722713x
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^conv1d_8/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall^gru_7/StatefulPartitionedCall%^simple_rnn_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2D
 conv1d_8/StatefulPartitionedCall conv1d_8/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2>
gru_7/StatefulPartitionedCallgru_7/StatefulPartitionedCall2L
$simple_rnn_6/StatefulPartitionedCall$simple_rnn_6/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_gru_7_layer_call_and_return_conditional_losses_2726327

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
 __inference_standard_gru_2726112i

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
:���������@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�-
�
while_body_2722853
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
�
)__inference_gpu_gru_with_fallback_2723019

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
:���������@P
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
4:��������� :��������� : :*
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
H:���������@:��������� :@`: `:`*<
api_implements*(gru_b5d7929a-c103-4256-9d6a-9fca617f4787*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:���������@
 
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
�
 __inference_standard_gru_2721536

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
while_body_2721446*
condR
while_cond_2721445*R
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
api_implements*(gru_298461b9-b437-48ed-afb0-0a379bc27613*
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
�
�
B__inference_gru_7_layer_call_and_return_conditional_losses_2722140

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
 __inference_standard_gru_2721925i

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
�
G
+__inference_flatten_9_layer_call_fn_2724654

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_2722184`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�-
�
while_body_2721835
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
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2722294

inputsC
1simple_rnn_cell_12_matmul_readvariableop_resource: @@
2simple_rnn_cell_12_biasadd_readvariableop_resource:@E
3simple_rnn_cell_12_matmul_1_readvariableop_resource:@@
identity��)simple_rnn_cell_12/BiasAdd/ReadVariableOp�(simple_rnn_cell_12/MatMul/ReadVariableOp�*simple_rnn_cell_12/MatMul_1/ReadVariableOp�while;
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
:��������� D
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
valueB"����    �
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
:��������� *
shrink_axis_mask�
(simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_12_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0�
simple_rnn_cell_12/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell_12/BiasAddBiasAdd#simple_rnn_cell_12/MatMul:product:01simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_12_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell_12/MatMul_1MatMulzeros:output:02simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell_12/addAddV2#simple_rnn_cell_12/BiasAdd:output:0%simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@m
simple_rnn_cell_12/TanhTanhsimple_rnn_cell_12/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_12_matmul_readvariableop_resource2simple_rnn_cell_12_biasadd_readvariableop_resource3simple_rnn_cell_12_matmul_1_readvariableop_resource*
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
while_body_2722228*
condR
while_cond_2722227*8
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
:���������@*
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
:���������@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp*^simple_rnn_cell_12/BiasAdd/ReadVariableOp)^simple_rnn_cell_12/MatMul/ReadVariableOp+^simple_rnn_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:��������� : : : 2V
)simple_rnn_cell_12/BiasAdd/ReadVariableOp)simple_rnn_cell_12/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_12/MatMul/ReadVariableOp(simple_rnn_cell_12/MatMul/ReadVariableOp2X
*simple_rnn_cell_12/MatMul_1/ReadVariableOp*simple_rnn_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
'__inference_gru_7_layer_call_fn_2725193

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
B__inference_gru_7_layer_call_and_return_conditional_losses_2723158o
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
:���������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�,
�
while_body_2724975
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_12_matmul_readvariableop_resource_0: @H
:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0:@M
;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_12_matmul_readvariableop_resource: @F
8while_simple_rnn_cell_12_biasadd_readvariableop_resource:@K
9while_simple_rnn_cell_12_matmul_1_readvariableop_resource:@@��/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp�.while/simple_rnn_cell_12/MatMul/ReadVariableOp�0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:��������� *
element_dtype0�
.while/simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_12_matmul_readvariableop_resource_0*
_output_shapes

: @*
dtype0�
while/simple_rnn_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/while/simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
 while/simple_rnn_cell_12/BiasAddBiasAdd)while/simple_rnn_cell_12/MatMul:product:07while/simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0while/simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
!while/simple_rnn_cell_12/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell_12/addAddV2)while/simple_rnn_cell_12/BiasAdd:output:0+while/simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@y
while/simple_rnn_cell_12/TanhTanh while/simple_rnn_cell_12/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_12/Tanh:y:0*
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
while/Identity_4Identity!while/simple_rnn_cell_12/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp0^while/simple_rnn_cell_12/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_12/MatMul/ReadVariableOp1^while/simple_rnn_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_12_biasadd_readvariableop_resource:while_simple_rnn_cell_12_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_12_matmul_1_readvariableop_resource;while_simple_rnn_cell_12_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_12_matmul_readvariableop_resource9while_simple_rnn_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2b
/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp/while/simple_rnn_cell_12/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_12/MatMul/ReadVariableOp.while/simple_rnn_cell_12/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp0while/simple_rnn_cell_12/MatMul_1/ReadVariableOp: 
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
܃
�

J__inference_sequential_20_layer_call_and_return_conditional_losses_2724624

inputsJ
4conv1d_8_conv1d_expanddims_1_readvariableop_resource: 6
(conv1d_8_biasadd_readvariableop_resource: P
>simple_rnn_6_simple_rnn_cell_12_matmul_readvariableop_resource: @M
?simple_rnn_6_simple_rnn_cell_12_biasadd_readvariableop_resource:@R
@simple_rnn_6_simple_rnn_cell_12_matmul_1_readvariableop_resource:@@4
"gru_7_read_readvariableop_resource:@`6
$gru_7_read_1_readvariableop_resource: `6
$gru_7_read_2_readvariableop_resource:`9
'dense_28_matmul_readvariableop_resource: 6
(dense_28_biasadd_readvariableop_resource:9
'dense_29_matmul_readvariableop_resource:6
(dense_29_biasadd_readvariableop_resource:
identity��conv1d_8/BiasAdd/ReadVariableOp�+conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp�dense_28/BiasAdd/ReadVariableOp�dense_28/MatMul/ReadVariableOp�dense_29/BiasAdd/ReadVariableOp�dense_29/MatMul/ReadVariableOp�gru_7/Read/ReadVariableOp�gru_7/Read_1/ReadVariableOp�gru_7/Read_2/ReadVariableOp�6simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOp�5simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOp�7simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOp�simple_rnn_6/whilei
conv1d_8/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d_8/Conv1D/ExpandDims
ExpandDimsinputs'conv1d_8/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
+conv1d_8/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_8_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0b
 conv1d_8/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d_8/Conv1D/ExpandDims_1
ExpandDims3conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_8/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: �
conv1d_8/Conv1DConv2D#conv1d_8/Conv1D/ExpandDims:output:0%conv1d_8/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
conv1d_8/Conv1D/SqueezeSqueezeconv1d_8/Conv1D:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

����������
conv1d_8/BiasAdd/ReadVariableOpReadVariableOp(conv1d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv1d_8/BiasAddBiasAdd conv1d_8/Conv1D/Squeeze:output:0'conv1d_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� f
conv1d_8/ReluReluconv1d_8/BiasAdd:output:0*
T0*+
_output_shapes
:��������� `
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    �
flatten_9/ReshapeReshapeconv1d_8/Relu:activations:0flatten_9/Const:output:0*
T0*'
_output_shapes
:��������� `
repeat_vector_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
repeat_vector_2/ExpandDims
ExpandDimsflatten_9/Reshape:output:0'repeat_vector_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:��������� j
repeat_vector_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"         �
repeat_vector_2/TileTile#repeat_vector_2/ExpandDims:output:0repeat_vector_2/stack:output:0*
T0*+
_output_shapes
:��������� _
simple_rnn_6/ShapeShaperepeat_vector_2/Tile:output:0*
T0*
_output_shapes
:j
 simple_rnn_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_6/strided_sliceStridedSlicesimple_rnn_6/Shape:output:0)simple_rnn_6/strided_slice/stack:output:0+simple_rnn_6/strided_slice/stack_1:output:0+simple_rnn_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@�
simple_rnn_6/zeros/packedPack#simple_rnn_6/strided_slice:output:0$simple_rnn_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
simple_rnn_6/zerosFill"simple_rnn_6/zeros/packed:output:0!simple_rnn_6/zeros/Const:output:0*
T0*'
_output_shapes
:���������@p
simple_rnn_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_6/transpose	Transposerepeat_vector_2/Tile:output:0$simple_rnn_6/transpose/perm:output:0*
T0*+
_output_shapes
:��������� ^
simple_rnn_6/Shape_1Shapesimple_rnn_6/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_6/strided_slice_1StridedSlicesimple_rnn_6/Shape_1:output:0+simple_rnn_6/strided_slice_1/stack:output:0-simple_rnn_6/strided_slice_1/stack_1:output:0-simple_rnn_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
simple_rnn_6/TensorArrayV2TensorListReserve1simple_rnn_6/TensorArrayV2/element_shape:output:0%simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Bsimple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
4simple_rnn_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_6/transpose:y:0Ksimple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
"simple_rnn_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_6/strided_slice_2StridedSlicesimple_rnn_6/transpose:y:0+simple_rnn_6/strided_slice_2/stack:output:0-simple_rnn_6/strided_slice_2/stack_1:output:0-simple_rnn_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask�
5simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOpReadVariableOp>simple_rnn_6_simple_rnn_cell_12_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0�
&simple_rnn_6/simple_rnn_cell_12/MatMulMatMul%simple_rnn_6/strided_slice_2:output:0=simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
6simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOpReadVariableOp?simple_rnn_6_simple_rnn_cell_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
'simple_rnn_6/simple_rnn_cell_12/BiasAddBiasAdd0simple_rnn_6/simple_rnn_cell_12/MatMul:product:0>simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
7simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOpReadVariableOp@simple_rnn_6_simple_rnn_cell_12_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
(simple_rnn_6/simple_rnn_cell_12/MatMul_1MatMulsimple_rnn_6/zeros:output:0?simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
#simple_rnn_6/simple_rnn_cell_12/addAddV20simple_rnn_6/simple_rnn_cell_12/BiasAdd:output:02simple_rnn_6/simple_rnn_cell_12/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
$simple_rnn_6/simple_rnn_cell_12/TanhTanh'simple_rnn_6/simple_rnn_cell_12/add:z:0*
T0*'
_output_shapes
:���������@{
*simple_rnn_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
simple_rnn_6/TensorArrayV2_1TensorListReserve3simple_rnn_6/TensorArrayV2_1/element_shape:output:0%simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���S
simple_rnn_6/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������a
simple_rnn_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
simple_rnn_6/whileWhile(simple_rnn_6/while/loop_counter:output:0.simple_rnn_6/while/maximum_iterations:output:0simple_rnn_6/time:output:0%simple_rnn_6/TensorArrayV2_1:handle:0simple_rnn_6/zeros:output:0%simple_rnn_6/strided_slice_1:output:0Dsimple_rnn_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0>simple_rnn_6_simple_rnn_cell_12_matmul_readvariableop_resource?simple_rnn_6_simple_rnn_cell_12_biasadd_readvariableop_resource@simple_rnn_6_simple_rnn_cell_12_matmul_1_readvariableop_resource*
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
simple_rnn_6_while_body_2724172*+
cond#R!
simple_rnn_6_while_cond_2724171*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
=simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
/simple_rnn_6/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_6/while:output:3Fsimple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0u
"simple_rnn_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������n
$simple_rnn_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_6/strided_slice_3StridedSlice8simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_6/strided_slice_3/stack:output:0-simple_rnn_6/strided_slice_3/stack_1:output:0-simple_rnn_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maskr
simple_rnn_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_6/transpose_1	Transpose8simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@W
gru_7/ShapeShapesimple_rnn_6/transpose_1:y:0*
T0*
_output_shapes
:c
gru_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gru_7/strided_sliceStridedSlicegru_7/Shape:output:0"gru_7/strided_slice/stack:output:0$gru_7/strided_slice/stack_1:output:0$gru_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
gru_7/zeros/packedPackgru_7/strided_slice:output:0gru_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_7/zerosFillgru_7/zeros/packed:output:0gru_7/zeros/Const:output:0*
T0*'
_output_shapes
:��������� |
gru_7/Read/ReadVariableOpReadVariableOp"gru_7_read_readvariableop_resource*
_output_shapes

:@`*
dtype0f
gru_7/IdentityIdentity!gru_7/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:@`�
gru_7/Read_1/ReadVariableOpReadVariableOp$gru_7_read_1_readvariableop_resource*
_output_shapes

: `*
dtype0j
gru_7/Identity_1Identity#gru_7/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `�
gru_7/Read_2/ReadVariableOpReadVariableOp$gru_7_read_2_readvariableop_resource*
_output_shapes

:`*
dtype0j
gru_7/Identity_2Identity#gru_7/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`�
gru_7/PartitionedCallPartitionedCallsimple_rnn_6/transpose_1:y:0gru_7/zeros:output:0gru_7/Identity:output:0gru_7/Identity_1:output:0gru_7/Identity_2:output:0*
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
 __inference_standard_gru_2724397�
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_28/MatMulMatMulgru_7/PartitionedCall:output:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_29/MatMulMatMuldense_28/BiasAdd:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_29/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^conv1d_8/BiasAdd/ReadVariableOp,^conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp^gru_7/Read/ReadVariableOp^gru_7/Read_1/ReadVariableOp^gru_7/Read_2/ReadVariableOp7^simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOp6^simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOp8^simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOp^simple_rnn_6/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2B
conv1d_8/BiasAdd/ReadVariableOpconv1d_8/BiasAdd/ReadVariableOp2Z
+conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp26
gru_7/Read/ReadVariableOpgru_7/Read/ReadVariableOp2:
gru_7/Read_1/ReadVariableOpgru_7/Read_1/ReadVariableOp2:
gru_7/Read_2/ReadVariableOpgru_7/Read_2/ReadVariableOp2p
6simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOp6simple_rnn_6/simple_rnn_cell_12/BiasAdd/ReadVariableOp2n
5simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOp5simple_rnn_6/simple_rnn_cell_12/MatMul/ReadVariableOp2r
7simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOp7simple_rnn_6/simple_rnn_cell_12/MatMul_1/ReadVariableOp2(
simple_rnn_6/whilesimple_rnn_6/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_sequential_20_layer_call_fn_2723600

inputs
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@@
	unknown_4:@`
	unknown_5: `
	unknown_6:`
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_20_layer_call_and_return_conditional_losses_2723379o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�"
�
J__inference_sequential_20_layer_call_and_return_conditional_losses_2723505
conv1d_8_input&
conv1d_8_2723473: 
conv1d_8_2723475: &
simple_rnn_6_2723480: @"
simple_rnn_6_2723482:@&
simple_rnn_6_2723484:@@
gru_7_2723487:@`
gru_7_2723489: `
gru_7_2723491:`"
dense_28_2723494: 
dense_28_2723496:"
dense_29_2723499:
dense_29_2723501:
identity�� conv1d_8/StatefulPartitionedCall� dense_28/StatefulPartitionedCall� dense_29/StatefulPartitionedCall�gru_7/StatefulPartitionedCall�$simple_rnn_6/StatefulPartitionedCall�
 conv1d_8/StatefulPartitionedCallStatefulPartitionedCallconv1d_8_inputconv1d_8_2723473conv1d_8_2723475*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv1d_8_layer_call_and_return_conditional_losses_2722172�
flatten_9/PartitionedCallPartitionedCall)conv1d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_2722184�
repeat_vector_2/PartitionedCallPartitionedCall"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_2721074�
$simple_rnn_6/StatefulPartitionedCallStatefulPartitionedCall(repeat_vector_2/PartitionedCall:output:0simple_rnn_6_2723480simple_rnn_6_2723482simple_rnn_6_2723484*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2723288�
gru_7/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_6/StatefulPartitionedCall:output:0gru_7_2723487gru_7_2723489gru_7_2723491*
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
B__inference_gru_7_layer_call_and_return_conditional_losses_2723158�
 dense_28/StatefulPartitionedCallStatefulPartitionedCall&gru_7/StatefulPartitionedCall:output:0dense_28_2723494dense_28_2723496*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_2722697�
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_2723499dense_29_2723501*
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
E__inference_dense_29_layer_call_and_return_conditional_losses_2722713x
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^conv1d_8/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall^gru_7/StatefulPartitionedCall%^simple_rnn_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2D
 conv1d_8/StatefulPartitionedCall conv1d_8/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2>
gru_7/StatefulPartitionedCallgru_7/StatefulPartitionedCall2L
$simple_rnn_6/StatefulPartitionedCall$simple_rnn_6/StatefulPartitionedCall:[ W
+
_output_shapes
:���������
(
_user_specified_nameconv1d_8_input
�>
�
 __inference_standard_gru_2725356

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
while_body_2725266*
condR
while_cond_2725265*R
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
api_implements*(gru_512f6324-f693-4328-96e4-47ae443c807d*
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

_user_specified_namebias"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
M
conv1d_8_input;
 serving_default_conv1d_8_input:0���������<
dense_290
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,cell
-
state_spec"
_tf_keras_rnn_layer
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4_random_generator
5cell
6
state_spec"
_tf_keras_rnn_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias"
_tf_keras_layer
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias"
_tf_keras_layer
v
0
1
G2
H3
I4
J5
K6
L7
=8
>9
E10
F11"
trackable_list_wrapper
v
0
1
G2
H3
I4
J5
K6
L7
=8
>9
E10
F11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Rtrace_0
Strace_1
Ttrace_2
Utrace_32�
/__inference_sequential_20_layer_call_fn_2722747
/__inference_sequential_20_layer_call_fn_2723571
/__inference_sequential_20_layer_call_fn_2723600
/__inference_sequential_20_layer_call_fn_2723435�
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
 zRtrace_0zStrace_1zTtrace_2zUtrace_3
�
Vtrace_0
Wtrace_1
Xtrace_2
Ytrace_32�
J__inference_sequential_20_layer_call_and_return_conditional_losses_2724112
J__inference_sequential_20_layer_call_and_return_conditional_losses_2724624
J__inference_sequential_20_layer_call_and_return_conditional_losses_2723470
J__inference_sequential_20_layer_call_and_return_conditional_losses_2723505�
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
 zVtrace_0zWtrace_1zXtrace_2zYtrace_3
�B�
"__inference__wrapped_model_2721062conv1d_8_input"�
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
�
Ziter

[beta_1

\beta_2
	]decay
^learning_ratem�m�=m�>m�Em�Fm�Gm�Hm�Im�Jm�Km�Lm�v�v�=v�>v�Ev�Fv�Gv�Hv�Iv�Jv�Kv�Lv�"
	optimizer
,
_serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
etrace_02�
*__inference_conv1d_8_layer_call_fn_2724633�
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
 zetrace_0
�
ftrace_02�
E__inference_conv1d_8_layer_call_and_return_conditional_losses_2724649�
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
 zftrace_0
%:# 2conv1d_8/kernel
: 2conv1d_8/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
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
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
ltrace_02�
+__inference_flatten_9_layer_call_fn_2724654�
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
 zltrace_0
�
mtrace_02�
F__inference_flatten_9_layer_call_and_return_conditional_losses_2724660�
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
 zmtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�
strace_02�
1__inference_repeat_vector_2_layer_call_fn_2724665�
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
 zstrace_0
�
ttrace_02�
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_2724673�
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
 zttrace_0
5
G0
H1
I2"
trackable_list_wrapper
5
G0
H1
I2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

ustates
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
�
{trace_0
|trace_1
}trace_2
~trace_32�
.__inference_simple_rnn_6_layer_call_fn_2724684
.__inference_simple_rnn_6_layer_call_fn_2724695
.__inference_simple_rnn_6_layer_call_fn_2724706
.__inference_simple_rnn_6_layer_call_fn_2724717�
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
 z{trace_0z|trace_1z}trace_2z~trace_3
�
trace_0
�trace_1
�trace_2
�trace_32�
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2724825
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2724933
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2725041
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2725149�
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
 ztrace_0z�trace_1z�trace_2z�trace_3
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator

Gkernel
Hrecurrent_kernel
Ibias"
_tf_keras_layer
 "
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�states
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
'__inference_gru_7_layer_call_fn_2725160
'__inference_gru_7_layer_call_fn_2725171
'__inference_gru_7_layer_call_fn_2725182
'__inference_gru_7_layer_call_fn_2725193�
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
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
B__inference_gru_7_layer_call_and_return_conditional_losses_2725571
B__inference_gru_7_layer_call_and_return_conditional_losses_2725949
B__inference_gru_7_layer_call_and_return_conditional_losses_2726327
B__inference_gru_7_layer_call_and_return_conditional_losses_2726705�
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
 z�trace_0z�trace_1z�trace_2z�trace_3
"
_generic_user_object
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator

Jkernel
Krecurrent_kernel
Lbias"
_tf_keras_layer
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_28_layer_call_fn_2726714�
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
 z�trace_0
�
�trace_02�
E__inference_dense_28_layer_call_and_return_conditional_losses_2726724�
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
 z�trace_0
!: 2dense_28/kernel
:2dense_28/bias
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_29_layer_call_fn_2726733�
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
 z�trace_0
�
�trace_02�
E__inference_dense_29_layer_call_and_return_conditional_losses_2726743�
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
 z�trace_0
!:2dense_29/kernel
:2dense_29/bias
8:6 @2&simple_rnn_6/simple_rnn_cell_12/kernel
B:@@@20simple_rnn_6/simple_rnn_cell_12/recurrent_kernel
2:0@2$simple_rnn_6/simple_rnn_cell_12/bias
*:(@`2gru_7/gru_cell_14/kernel
4:2 `2"gru_7/gru_cell_14/recurrent_kernel
(:&`2gru_7/gru_cell_14/bias
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_sequential_20_layer_call_fn_2722747conv1d_8_input"�
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
/__inference_sequential_20_layer_call_fn_2723571inputs"�
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
/__inference_sequential_20_layer_call_fn_2723600inputs"�
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
/__inference_sequential_20_layer_call_fn_2723435conv1d_8_input"�
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
J__inference_sequential_20_layer_call_and_return_conditional_losses_2724112inputs"�
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
J__inference_sequential_20_layer_call_and_return_conditional_losses_2724624inputs"�
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
J__inference_sequential_20_layer_call_and_return_conditional_losses_2723470conv1d_8_input"�
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
J__inference_sequential_20_layer_call_and_return_conditional_losses_2723505conv1d_8_input"�
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
%__inference_signature_wrapper_2723542conv1d_8_input"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv1d_8_layer_call_fn_2724633inputs"�
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
E__inference_conv1d_8_layer_call_and_return_conditional_losses_2724649inputs"�
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
+__inference_flatten_9_layer_call_fn_2724654inputs"�
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
F__inference_flatten_9_layer_call_and_return_conditional_losses_2724660inputs"�
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
1__inference_repeat_vector_2_layer_call_fn_2724665inputs"�
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
�B�
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_2724673inputs"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
,0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_simple_rnn_6_layer_call_fn_2724684inputs/0"�
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
.__inference_simple_rnn_6_layer_call_fn_2724695inputs/0"�
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
.__inference_simple_rnn_6_layer_call_fn_2724706inputs"�
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
.__inference_simple_rnn_6_layer_call_fn_2724717inputs"�
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
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2724825inputs/0"�
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
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2724933inputs/0"�
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
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2725041inputs"�
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
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2725149inputs"�
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
G0
H1
I2"
trackable_list_wrapper
5
G0
H1
I2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_simple_rnn_cell_12_layer_call_fn_2726757
4__inference_simple_rnn_cell_12_layer_call_fn_2726771�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_simple_rnn_cell_12_layer_call_and_return_conditional_losses_2726788
O__inference_simple_rnn_cell_12_layer_call_and_return_conditional_losses_2726805�
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
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
50"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_gru_7_layer_call_fn_2725160inputs/0"�
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
'__inference_gru_7_layer_call_fn_2725171inputs/0"�
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
'__inference_gru_7_layer_call_fn_2725182inputs"�
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
'__inference_gru_7_layer_call_fn_2725193inputs"�
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
B__inference_gru_7_layer_call_and_return_conditional_losses_2725571inputs/0"�
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
B__inference_gru_7_layer_call_and_return_conditional_losses_2725949inputs/0"�
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
B__inference_gru_7_layer_call_and_return_conditional_losses_2726327inputs"�
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
B__inference_gru_7_layer_call_and_return_conditional_losses_2726705inputs"�
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
J0
K1
L2"
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
*__inference_dense_28_layer_call_fn_2726714inputs"�
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
E__inference_dense_28_layer_call_and_return_conditional_losses_2726724inputs"�
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
*__inference_dense_29_layer_call_fn_2726733inputs"�
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
E__inference_dense_29_layer_call_and_return_conditional_losses_2726743inputs"�
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
4__inference_simple_rnn_cell_12_layer_call_fn_2726757inputsstates/0"�
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
4__inference_simple_rnn_cell_12_layer_call_fn_2726771inputsstates/0"�
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
O__inference_simple_rnn_cell_12_layer_call_and_return_conditional_losses_2726788inputsstates/0"�
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
O__inference_simple_rnn_cell_12_layer_call_and_return_conditional_losses_2726805inputsstates/0"�
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
*:( 2Adam/conv1d_8/kernel/m
 : 2Adam/conv1d_8/bias/m
&:$ 2Adam/dense_28/kernel/m
 :2Adam/dense_28/bias/m
&:$2Adam/dense_29/kernel/m
 :2Adam/dense_29/bias/m
=:; @2-Adam/simple_rnn_6/simple_rnn_cell_12/kernel/m
G:E@@27Adam/simple_rnn_6/simple_rnn_cell_12/recurrent_kernel/m
7:5@2+Adam/simple_rnn_6/simple_rnn_cell_12/bias/m
/:-@`2Adam/gru_7/gru_cell_14/kernel/m
9:7 `2)Adam/gru_7/gru_cell_14/recurrent_kernel/m
-:+`2Adam/gru_7/gru_cell_14/bias/m
*:( 2Adam/conv1d_8/kernel/v
 : 2Adam/conv1d_8/bias/v
&:$ 2Adam/dense_28/kernel/v
 :2Adam/dense_28/bias/v
&:$2Adam/dense_29/kernel/v
 :2Adam/dense_29/bias/v
=:; @2-Adam/simple_rnn_6/simple_rnn_cell_12/kernel/v
G:E@@27Adam/simple_rnn_6/simple_rnn_cell_12/recurrent_kernel/v
7:5@2+Adam/simple_rnn_6/simple_rnn_cell_12/bias/v
/:-@`2Adam/gru_7/gru_cell_14/kernel/v
9:7 `2)Adam/gru_7/gru_cell_14/recurrent_kernel/v
-:+`2Adam/gru_7/gru_cell_14/bias/v�
"__inference__wrapped_model_2721062�GIHJKL=>EF;�8
1�.
,�)
conv1d_8_input���������
� "3�0
.
dense_29"�
dense_29����������
E__inference_conv1d_8_layer_call_and_return_conditional_losses_2724649d3�0
)�&
$�!
inputs���������
� ")�&
�
0��������� 
� �
*__inference_conv1d_8_layer_call_fn_2724633W3�0
)�&
$�!
inputs���������
� "���������� �
E__inference_dense_28_layer_call_and_return_conditional_losses_2726724\=>/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� }
*__inference_dense_28_layer_call_fn_2726714O=>/�,
%�"
 �
inputs��������� 
� "�����������
E__inference_dense_29_layer_call_and_return_conditional_losses_2726743\EF/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� }
*__inference_dense_29_layer_call_fn_2726733OEF/�,
%�"
 �
inputs���������
� "�����������
F__inference_flatten_9_layer_call_and_return_conditional_losses_2724660\3�0
)�&
$�!
inputs��������� 
� "%�"
�
0��������� 
� ~
+__inference_flatten_9_layer_call_fn_2724654O3�0
)�&
$�!
inputs��������� 
� "���������� �
B__inference_gru_7_layer_call_and_return_conditional_losses_2725571}JKLO�L
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
B__inference_gru_7_layer_call_and_return_conditional_losses_2725949}JKLO�L
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
B__inference_gru_7_layer_call_and_return_conditional_losses_2726327mJKL?�<
5�2
$�!
inputs���������@

 
p 

 
� "%�"
�
0��������� 
� �
B__inference_gru_7_layer_call_and_return_conditional_losses_2726705mJKL?�<
5�2
$�!
inputs���������@

 
p

 
� "%�"
�
0��������� 
� �
'__inference_gru_7_layer_call_fn_2725160pJKLO�L
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
'__inference_gru_7_layer_call_fn_2725171pJKLO�L
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
'__inference_gru_7_layer_call_fn_2725182`JKL?�<
5�2
$�!
inputs���������@

 
p 

 
� "���������� �
'__inference_gru_7_layer_call_fn_2725193`JKL?�<
5�2
$�!
inputs���������@

 
p

 
� "���������� �
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_2724673n8�5
.�+
)�&
inputs������������������
� "2�/
(�%
0������������������
� �
1__inference_repeat_vector_2_layer_call_fn_2724665a8�5
.�+
)�&
inputs������������������
� "%�"�������������������
J__inference_sequential_20_layer_call_and_return_conditional_losses_2723470zGIHJKL=>EFC�@
9�6
,�)
conv1d_8_input���������
p 

 
� "%�"
�
0���������
� �
J__inference_sequential_20_layer_call_and_return_conditional_losses_2723505zGIHJKL=>EFC�@
9�6
,�)
conv1d_8_input���������
p

 
� "%�"
�
0���������
� �
J__inference_sequential_20_layer_call_and_return_conditional_losses_2724112rGIHJKL=>EF;�8
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
J__inference_sequential_20_layer_call_and_return_conditional_losses_2724624rGIHJKL=>EF;�8
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
/__inference_sequential_20_layer_call_fn_2722747mGIHJKL=>EFC�@
9�6
,�)
conv1d_8_input���������
p 

 
� "�����������
/__inference_sequential_20_layer_call_fn_2723435mGIHJKL=>EFC�@
9�6
,�)
conv1d_8_input���������
p

 
� "�����������
/__inference_sequential_20_layer_call_fn_2723571eGIHJKL=>EF;�8
1�.
$�!
inputs���������
p 

 
� "�����������
/__inference_sequential_20_layer_call_fn_2723600eGIHJKL=>EF;�8
1�.
$�!
inputs���������
p

 
� "�����������
%__inference_signature_wrapper_2723542�GIHJKL=>EFM�J
� 
C�@
>
conv1d_8_input,�)
conv1d_8_input���������"3�0
.
dense_29"�
dense_29����������
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2724825�GIHO�L
E�B
4�1
/�,
inputs/0������������������ 

 
p 

 
� "2�/
(�%
0������������������@
� �
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2724933�GIHO�L
E�B
4�1
/�,
inputs/0������������������ 

 
p

 
� "2�/
(�%
0������������������@
� �
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2725041qGIH?�<
5�2
$�!
inputs��������� 

 
p 

 
� ")�&
�
0���������@
� �
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_2725149qGIH?�<
5�2
$�!
inputs��������� 

 
p

 
� ")�&
�
0���������@
� �
.__inference_simple_rnn_6_layer_call_fn_2724684}GIHO�L
E�B
4�1
/�,
inputs/0������������������ 

 
p 

 
� "%�"������������������@�
.__inference_simple_rnn_6_layer_call_fn_2724695}GIHO�L
E�B
4�1
/�,
inputs/0������������������ 

 
p

 
� "%�"������������������@�
.__inference_simple_rnn_6_layer_call_fn_2724706dGIH?�<
5�2
$�!
inputs��������� 

 
p 

 
� "����������@�
.__inference_simple_rnn_6_layer_call_fn_2724717dGIH?�<
5�2
$�!
inputs��������� 

 
p

 
� "����������@�
O__inference_simple_rnn_cell_12_layer_call_and_return_conditional_losses_2726788�GIH\�Y
R�O
 �
inputs��������� 
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
O__inference_simple_rnn_cell_12_layer_call_and_return_conditional_losses_2726805�GIH\�Y
R�O
 �
inputs��������� 
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
4__inference_simple_rnn_cell_12_layer_call_fn_2726757�GIH\�Y
R�O
 �
inputs��������� 
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
4__inference_simple_rnn_cell_12_layer_call_fn_2726771�GIH\�Y
R�O
 �
inputs��������� 
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