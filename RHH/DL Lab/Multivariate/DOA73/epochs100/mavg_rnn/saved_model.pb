Й≤
Чз
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
Ѕ
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
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ч
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
∞
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleКйиelement_dtype"
element_dtypetype"

shape_typetype:
2	
Я
TensorListReserve
element_shape"
shape_type
num_elements(
handleКйиelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint€€€€€€€€€
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
И"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ек
ђ
*Adam/simple_rnn_2/simple_rnn_cell_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/simple_rnn_2/simple_rnn_cell_2/bias/v
•
>Adam/simple_rnn_2/simple_rnn_cell_2/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_2/simple_rnn_cell_2/bias/v*
_output_shapes
:@*
dtype0
»
6Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*G
shared_name86Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/v
Ѕ
JAdam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
і
,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*=
shared_name.,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/v
≠
@Adam/simple_rnn_2/simple_rnn_cell_2/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/v*
_output_shapes

:@*
dtype0
А
Adam/dense_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_32/bias/v
y
(Adam/dense_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/v*
_output_shapes
:*
dtype0
И
Adam/dense_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_32/kernel/v
Б
*Adam/dense_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/v*
_output_shapes

:*
dtype0
А
Adam/dense_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_31/bias/v
y
(Adam/dense_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/v*
_output_shapes
:*
dtype0
И
Adam/dense_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_31/kernel/v
Б
*Adam/dense_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/v*
_output_shapes

:@*
dtype0
ђ
*Adam/simple_rnn_2/simple_rnn_cell_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/simple_rnn_2/simple_rnn_cell_2/bias/m
•
>Adam/simple_rnn_2/simple_rnn_cell_2/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_2/simple_rnn_cell_2/bias/m*
_output_shapes
:@*
dtype0
»
6Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*G
shared_name86Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/m
Ѕ
JAdam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
і
,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*=
shared_name.,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/m
≠
@Adam/simple_rnn_2/simple_rnn_cell_2/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/m*
_output_shapes

:@*
dtype0
А
Adam/dense_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_32/bias/m
y
(Adam/dense_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/m*
_output_shapes
:*
dtype0
И
Adam/dense_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_32/kernel/m
Б
*Adam/dense_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/m*
_output_shapes

:*
dtype0
А
Adam/dense_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_31/bias/m
y
(Adam/dense_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/m*
_output_shapes
:*
dtype0
И
Adam/dense_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_31/kernel/m
Б
*Adam/dense_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/m*
_output_shapes

:@*
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
#simple_rnn_2/simple_rnn_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#simple_rnn_2/simple_rnn_cell_2/bias
Ч
7simple_rnn_2/simple_rnn_cell_2/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_2/simple_rnn_cell_2/bias*
_output_shapes
:@*
dtype0
Ї
/simple_rnn_2/simple_rnn_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*@
shared_name1/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel
≥
Csimple_rnn_2/simple_rnn_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel*
_output_shapes

:@@*
dtype0
¶
%simple_rnn_2/simple_rnn_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*6
shared_name'%simple_rnn_2/simple_rnn_cell_2/kernel
Я
9simple_rnn_2/simple_rnn_cell_2/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_2/simple_rnn_cell_2/kernel*
_output_shapes

:@*
dtype0
r
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_32/bias
k
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes
:*
dtype0
z
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_32/kernel
s
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel*
_output_shapes

:*
dtype0
r
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_31/bias
k
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes
:*
dtype0
z
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_31/kernel
s
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel*
_output_shapes

:@*
dtype0
Н
"serving_default_simple_rnn_2_inputPlaceholder*+
_output_shapes
:€€€€€€€€€*
dtype0* 
shape:€€€€€€€€€
Й
StatefulPartitionedCallStatefulPartitionedCall"serving_default_simple_rnn_2_input%simple_rnn_2/simple_rnn_cell_2/kernel#simple_rnn_2/simple_rnn_cell_2/bias/simple_rnn_2/simple_rnn_cell_2/recurrent_kerneldense_31/kerneldense_31/biasdense_32/kerneldense_32/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_1535711

NoOpNoOp
т8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*≠8
value£8B†8 BЩ8
Ѕ
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
™
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec*
¶
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¶
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*
5
%0
&1
'2
3
4
#5
$6*
5
%0
&1
'2
3
4
#5
$6*
* 
∞
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
-trace_0
.trace_1
/trace_2
0trace_3* 
6
1trace_0
2trace_1
3trace_2
4trace_3* 
* 
¬
5iter

6beta_1

7beta_2
	8decay
9learning_ratemqmr#ms$mt%mu&mv'mwvxvy#vz$v{%v|&v}'v~*

:serving_default* 

%0
&1
'2*

%0
&1
'2*
* 
Я

;states
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Atrace_0
Btrace_1
Ctrace_2
Dtrace_3* 
6
Etrace_0
Ftrace_1
Gtrace_2
Htrace_3* 
”
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
O_random_generator

%kernel
&recurrent_kernel
'bias*
* 

0
1*

0
1*
* 
У
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Utrace_0* 

Vtrace_0* 
_Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_31/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

#0
$1*

#0
$1*
* 
У
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

\trace_0* 

]trace_0* 
_Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_32/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_2/simple_rnn_cell_2/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#simple_rnn_2/simple_rnn_cell_2/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

^0
_1*
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
%0
&1
'2*

%0
&1
'2*
* 
У
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

etrace_0
ftrace_1* 

gtrace_0
htrace_1* 
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
i	variables
j	keras_api
	ktotal
	lcount*
8
m	variables
n	keras_api
	ototal
	pcount*
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
k0
l1*

i	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

o0
p1*

m	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_31/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_31/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_32/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_32/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
УМ
VARIABLE_VALUE6Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUE*Adam/simple_rnn_2/simple_rnn_cell_2/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_31/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_31/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_32/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_32/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
УМ
VARIABLE_VALUE6Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUE*Adam/simple_rnn_2/simple_rnn_cell_2/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Р
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp9simple_rnn_2/simple_rnn_cell_2/kernel/Read/ReadVariableOpCsimple_rnn_2/simple_rnn_cell_2/recurrent_kernel/Read/ReadVariableOp7simple_rnn_2/simple_rnn_cell_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_31/kernel/m/Read/ReadVariableOp(Adam/dense_31/bias/m/Read/ReadVariableOp*Adam/dense_32/kernel/m/Read/ReadVariableOp(Adam/dense_32/bias/m/Read/ReadVariableOp@Adam/simple_rnn_2/simple_rnn_cell_2/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_2/simple_rnn_cell_2/bias/m/Read/ReadVariableOp*Adam/dense_31/kernel/v/Read/ReadVariableOp(Adam/dense_31/bias/v/Read/ReadVariableOp*Adam/dense_32/kernel/v/Read/ReadVariableOp(Adam/dense_32/bias/v/Read/ReadVariableOp@Adam/simple_rnn_2/simple_rnn_cell_2/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_2/simple_rnn_cell_2/bias/v/Read/ReadVariableOpConst*+
Tin$
"2 	*
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
 __inference__traced_save_1536690
≥
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_31/kerneldense_31/biasdense_32/kerneldense_32/bias%simple_rnn_2/simple_rnn_cell_2/kernel/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel#simple_rnn_2/simple_rnn_cell_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_31/kernel/mAdam/dense_31/bias/mAdam/dense_32/kernel/mAdam/dense_32/bias/m,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/m6Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/m*Adam/simple_rnn_2/simple_rnn_cell_2/bias/mAdam/dense_31/kernel/vAdam/dense_31/bias/vAdam/dense_32/kernel/vAdam/dense_32/bias/v,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/v6Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/v*Adam/simple_rnn_2/simple_rnn_cell_2/bias/v**
Tin#
!2*
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
#__inference__traced_restore_1536790Э„
я
ѓ
while_cond_1536409
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1536409___redundant_placeholder05
1while_while_cond_1536409___redundant_placeholder15
1while_while_cond_1536409___redundant_placeholder25
1while_while_cond_1536409___redundant_placeholder3
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
-: : : : :€€€€€€€€€@: ::::: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
я
ѓ
while_cond_1536299
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1536299___redundant_placeholder05
1while_while_cond_1536299___redundant_placeholder15
1while_while_cond_1536299___redundant_placeholder25
1while_while_cond_1536299___redundant_placeholder3
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
-: : : : :€€€€€€€€€@: ::::: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
ќ	
±
/__inference_sequential_16_layer_call_fn_1535412
simple_rnn_2_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:
	unknown_4:
	unknown_5:
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535395o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:€€€€€€€€€
,
_user_specified_namesimple_rnn_2_input
ѓ
ё
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535663
simple_rnn_2_input&
simple_rnn_2_1535645:@"
simple_rnn_2_1535647:@&
simple_rnn_2_1535649:@@"
dense_31_1535652:@
dense_31_1535654:"
dense_32_1535657:
dense_32_1535659:
identityИҐ dense_31/StatefulPartitionedCallҐ dense_32/StatefulPartitionedCallҐ$simple_rnn_2/StatefulPartitionedCallІ
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputsimple_rnn_2_1535645simple_rnn_2_1535647simple_rnn_2_1535649*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1535354Ъ
 dense_31/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0dense_31_1535652dense_31_1535654*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_1535372Ц
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0dense_32_1535657dense_32_1535659*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_1535388x
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≥
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall:_ [
+
_output_shapes
:€€€€€€€€€
,
_user_specified_namesimple_rnn_2_input
Ш
Ї
.__inference_simple_rnn_2_layer_call_fn_1536004
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@@
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1535068o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
ƒ
Ч
*__inference_dense_32_layer_call_fn_1536505

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_1535388o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
«-
…
while_body_1536190
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:@G
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_2_matmul_readvariableop_resource:@E
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:@@ИҐ.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_2/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0¶
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0√
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@§
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Њ
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@™
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0™
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@ђ
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@w
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0 while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@я

while/NoOpNoOp/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€@: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
ї

џ
3__inference_simple_rnn_cell_2_layer_call_fn_1536529

inputs
states_0
unknown:@
	unknown_0:@
	unknown_1:@@
identity

identity_1ИҐStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_1534990o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0
ќ>
Њ
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1536147
inputs_0B
0simple_rnn_cell_2_matmul_readvariableop_resource:@?
1simple_rnn_cell_2_biasadd_readvariableop_resource:@D
2simple_rnn_cell_2_matmul_1_readvariableop_resource:@@
identityИҐ(simple_rnn_cell_2/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_2/MatMul/ReadVariableOpҐ)simple_rnn_cell_2/MatMul_1/ReadVariableOpҐwhile=
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
valueB:—
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
:€€€€€€€€€@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskШ
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Я
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ц
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ђ
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ь
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Щ
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ъ
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@k
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :≈
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1536080*
condR
while_cond_1536079*8
output_shapes'
%: : : : :€€€€€€€€€@: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
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
:€€€€€€€€€@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@ѕ
NoOpNoOp)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
Ђ>
Љ
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1536477

inputsB
0simple_rnn_cell_2_matmul_readvariableop_resource:@?
1simple_rnn_cell_2_biasadd_readvariableop_resource:@D
2simple_rnn_cell_2_matmul_1_readvariableop_resource:@@
identityИҐ(simple_rnn_cell_2/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_2/MatMul/ReadVariableOpҐ)simple_rnn_cell_2/MatMul_1/ReadVariableOpҐwhile;
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
valueB:—
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
:€€€€€€€€€@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskШ
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Я
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ц
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ђ
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ь
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Щ
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ъ
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@k
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :≈
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1536410*
condR
while_cond_1536409*8
output_shapes'
%: : : : :€€€€€€€€€@: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
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
:€€€€€€€€€@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@ѕ
NoOpNoOp)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
й9
ѕ
simple_rnn_2_while_body_15357926
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_25
1simple_rnn_2_while_simple_rnn_2_strided_slice_1_0q
msimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0:@T
Fsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:@Y
Gsimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:@@
simple_rnn_2_while_identity!
simple_rnn_2_while_identity_1!
simple_rnn_2_while_identity_2!
simple_rnn_2_while_identity_3!
simple_rnn_2_while_identity_43
/simple_rnn_2_while_simple_rnn_2_strided_slice_1o
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource:@R
Dsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource:@W
Esimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource:@@ИҐ;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpҐ:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOpҐ<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpХ
Dsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   з
6simple_rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_2_while_placeholderMsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0ј
:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0к
+simple_rnn_2/while/simple_rnn_cell_2/MatMulMatMul=simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Њ
;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0е
,simple_rnn_2/while/simple_rnn_cell_2/BiasAddBiasAdd5simple_rnn_2/while/simple_rnn_cell_2/MatMul:product:0Csimple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@ƒ
<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0—
-simple_rnn_2/while/simple_rnn_cell_2/MatMul_1MatMul simple_rnn_2_while_placeholder_2Dsimple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@”
(simple_rnn_2/while/simple_rnn_cell_2/addAddV25simple_rnn_2/while/simple_rnn_cell_2/BiasAdd:output:07simple_rnn_2/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@С
)simple_rnn_2/while/simple_rnn_cell_2/TanhTanh,simple_rnn_2/while/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@
=simple_rnn_2/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : •
7simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_2_while_placeholder_1Fsimple_rnn_2/while/TensorArrayV2Write/TensorListSetItem/index:output:0-simple_rnn_2/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype0:йи“Z
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
: ≠
simple_rnn_2/while/Identity_3IdentityGsimple_rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: §
simple_rnn_2/while/Identity_4Identity-simple_rnn_2/while/simple_rnn_cell_2/Tanh:y:0^simple_rnn_2/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@У
simple_rnn_2/while/NoOpNoOp<^simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0"G
simple_rnn_2_while_identity_1&simple_rnn_2/while/Identity_1:output:0"G
simple_rnn_2_while_identity_2&simple_rnn_2/while/Identity_2:output:0"G
simple_rnn_2_while_identity_3&simple_rnn_2/while/Identity_3:output:0"G
simple_rnn_2_while_identity_4&simple_rnn_2/while/Identity_4:output:0"d
/simple_rnn_2_while_simple_rnn_2_strided_slice_11simple_rnn_2_while_simple_rnn_2_strided_slice_1_0"О
Dsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resourceFsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"Р
Esimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resourceGsimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"М
Csimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resourceEsimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"№
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensormsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€@: : : : : 2z
;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2x
:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp2|
<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
™	
•
/__inference_sequential_16_layer_call_fn_1535749

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:
	unknown_4:
	unknown_5:
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535606o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
«-
…
while_body_1536300
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:@G
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_2_matmul_readvariableop_resource:@E
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:@@ИҐ.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_2/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0¶
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0√
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@§
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Њ
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@™
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0™
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@ђ
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@w
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0 while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@я

while/NoOpNoOp/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€@: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
Ђ>
Љ
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1535555

inputsB
0simple_rnn_cell_2_matmul_readvariableop_resource:@?
1simple_rnn_cell_2_biasadd_readvariableop_resource:@D
2simple_rnn_cell_2_matmul_1_readvariableop_resource:@@
identityИҐ(simple_rnn_cell_2/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_2/MatMul/ReadVariableOpҐ)simple_rnn_cell_2/MatMul_1/ReadVariableOpҐwhile;
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
valueB:—
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
:€€€€€€€€€@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskШ
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Я
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ц
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ђ
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ь
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Щ
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ъ
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@k
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :≈
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1535488*
condR
while_cond_1535487*8
output_shapes'
%: : : : :€€€€€€€€€@: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
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
:€€€€€€€€€@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@ѕ
NoOpNoOp)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ш
Ї
.__inference_simple_rnn_2_layer_call_fn_1536015
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@@
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1535229o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
«-
…
while_body_1535488
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:@G
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_2_matmul_readvariableop_resource:@E
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:@@ИҐ.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_2/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0¶
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0√
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@§
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Њ
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@™
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0™
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@ђ
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@w
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0 while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@я

while/NoOpNoOp/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€@: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
й9
ѕ
simple_rnn_2_while_body_15359146
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_25
1simple_rnn_2_while_simple_rnn_2_strided_slice_1_0q
msimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0:@T
Fsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:@Y
Gsimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:@@
simple_rnn_2_while_identity!
simple_rnn_2_while_identity_1!
simple_rnn_2_while_identity_2!
simple_rnn_2_while_identity_3!
simple_rnn_2_while_identity_43
/simple_rnn_2_while_simple_rnn_2_strided_slice_1o
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource:@R
Dsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource:@W
Esimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource:@@ИҐ;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpҐ:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOpҐ<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpХ
Dsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   з
6simple_rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_2_while_placeholderMsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0ј
:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0к
+simple_rnn_2/while/simple_rnn_cell_2/MatMulMatMul=simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Њ
;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0е
,simple_rnn_2/while/simple_rnn_cell_2/BiasAddBiasAdd5simple_rnn_2/while/simple_rnn_cell_2/MatMul:product:0Csimple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@ƒ
<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0—
-simple_rnn_2/while/simple_rnn_cell_2/MatMul_1MatMul simple_rnn_2_while_placeholder_2Dsimple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@”
(simple_rnn_2/while/simple_rnn_cell_2/addAddV25simple_rnn_2/while/simple_rnn_cell_2/BiasAdd:output:07simple_rnn_2/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@С
)simple_rnn_2/while/simple_rnn_cell_2/TanhTanh,simple_rnn_2/while/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@
=simple_rnn_2/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : •
7simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_2_while_placeholder_1Fsimple_rnn_2/while/TensorArrayV2Write/TensorListSetItem/index:output:0-simple_rnn_2/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype0:йи“Z
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
: ≠
simple_rnn_2/while/Identity_3IdentityGsimple_rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: §
simple_rnn_2/while/Identity_4Identity-simple_rnn_2/while/simple_rnn_cell_2/Tanh:y:0^simple_rnn_2/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@У
simple_rnn_2/while/NoOpNoOp<^simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0"G
simple_rnn_2_while_identity_1&simple_rnn_2/while/Identity_1:output:0"G
simple_rnn_2_while_identity_2&simple_rnn_2/while/Identity_2:output:0"G
simple_rnn_2_while_identity_3&simple_rnn_2/while/Identity_3:output:0"G
simple_rnn_2_while_identity_4&simple_rnn_2/while/Identity_4:output:0"d
/simple_rnn_2_while_simple_rnn_2_strided_slice_11simple_rnn_2_while_simple_rnn_2_strided_slice_1_0"О
Dsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resourceFsimple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"Р
Esimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resourceGsimple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"М
Csimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resourceEsimple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"№
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensormsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€@: : : : : 2z
;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp;simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2x
:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp:simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp2|
<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp<simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
»	
ц
E__inference_dense_31_layer_call_and_return_conditional_losses_1535372

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
я
ѓ
while_cond_1535003
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1535003___redundant_placeholder05
1while_while_cond_1535003___redundant_placeholder15
1while_while_cond_1535003___redundant_placeholder25
1while_while_cond_1535003___redundant_placeholder3
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
-: : : : :€€€€€€€€€@: ::::: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
ї

џ
3__inference_simple_rnn_cell_2_layer_call_fn_1536543

inputs
states_0
unknown:@
	unknown_0:@
	unknown_1:@@
identity

identity_1ИҐStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_1535112o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0
я
ѓ
while_cond_1536079
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1536079___redundant_placeholder05
1while_while_cond_1536079___redundant_placeholder15
1while_while_cond_1536079___redundant_placeholder25
1while_while_cond_1536079___redundant_placeholder3
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
-: : : : :€€€€€€€€€@: ::::: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
ю
й
N__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_1535112

inputs

states0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates
»	
ц
E__inference_dense_32_layer_call_and_return_conditional_losses_1535388

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
я
ѓ
while_cond_1535487
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1535487___redundant_placeholder05
1while_while_cond_1535487___redundant_placeholder15
1while_while_cond_1535487___redundant_placeholder25
1while_while_cond_1535487___redundant_placeholder3
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
-: : : : :€€€€€€€€€@: ::::: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
Е5
Я
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1535068

inputs+
simple_rnn_cell_2_1534991:@'
simple_rnn_cell_2_1534993:@+
simple_rnn_cell_2_1534995:@@
identityИҐ)simple_rnn_cell_2/StatefulPartitionedCallҐwhile;
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
valueB:—
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
:€€€€€€€€€@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskл
)simple_rnn_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_2_1534991simple_rnn_cell_2_1534993simple_rnn_cell_2_1534995*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_1534990n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :≈
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Т
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_2_1534991simple_rnn_cell_2_1534993simple_rnn_cell_2_1534995*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1535004*
condR
while_cond_1535003*8
output_shapes'
%: : : : :€€€€€€€€€@: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
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
:€€€€€€€€€@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@z
NoOpNoOp*^simple_rnn_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2V
)simple_rnn_cell_2/StatefulPartitionedCall)simple_rnn_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
¬C
Н
 __inference__traced_save_1536690
file_prefix.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableopD
@savev2_simple_rnn_2_simple_rnn_cell_2_kernel_read_readvariableopN
Jsavev2_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_2_simple_rnn_cell_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_31_kernel_m_read_readvariableop3
/savev2_adam_dense_31_bias_m_read_readvariableop5
1savev2_adam_dense_32_kernel_m_read_readvariableop3
/savev2_adam_dense_32_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_2_simple_rnn_cell_2_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_2_simple_rnn_cell_2_bias_m_read_readvariableop5
1savev2_adam_dense_31_kernel_v_read_readvariableop3
/savev2_adam_dense_31_bias_v_read_readvariableop5
1savev2_adam_dense_32_kernel_v_read_readvariableop3
/savev2_adam_dense_32_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_2_simple_rnn_cell_2_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_2_simple_rnn_cell_2_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
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
: Ћ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ф
valueкBзB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЂ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ц
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop@savev2_simple_rnn_2_simple_rnn_cell_2_kernel_read_readvariableopJsavev2_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_read_readvariableop>savev2_simple_rnn_2_simple_rnn_cell_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_31_kernel_m_read_readvariableop/savev2_adam_dense_31_bias_m_read_readvariableop1savev2_adam_dense_32_kernel_m_read_readvariableop/savev2_adam_dense_32_bias_m_read_readvariableopGsavev2_adam_simple_rnn_2_simple_rnn_cell_2_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_2_simple_rnn_cell_2_bias_m_read_readvariableop1savev2_adam_dense_31_kernel_v_read_readvariableop/savev2_adam_dense_31_bias_v_read_readvariableop1savev2_adam_dense_32_kernel_v_read_readvariableop/savev2_adam_dense_32_bias_v_read_readvariableopGsavev2_adam_simple_rnn_2_simple_rnn_cell_2_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_2_simple_rnn_cell_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *-
dtypes#
!2	Р
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

identity_1Identity_1:output:0*ў
_input_shapes«
ƒ: :@::::@:@@:@: : : : : : : : : :@::::@:@@:@:@::::@:@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:
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
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:

_output_shapes
: 
цz
ћ
#__inference__traced_restore_1536790
file_prefix2
 assignvariableop_dense_31_kernel:@.
 assignvariableop_1_dense_31_bias:4
"assignvariableop_2_dense_32_kernel:.
 assignvariableop_3_dense_32_bias:J
8assignvariableop_4_simple_rnn_2_simple_rnn_cell_2_kernel:@T
Bassignvariableop_5_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel:@@D
6assignvariableop_6_simple_rnn_2_simple_rnn_cell_2_bias:@&
assignvariableop_7_adam_iter:	 (
assignvariableop_8_adam_beta_1: (
assignvariableop_9_adam_beta_2: (
assignvariableop_10_adam_decay: 0
&assignvariableop_11_adam_learning_rate: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: #
assignvariableop_14_total: #
assignvariableop_15_count: <
*assignvariableop_16_adam_dense_31_kernel_m:@6
(assignvariableop_17_adam_dense_31_bias_m:<
*assignvariableop_18_adam_dense_32_kernel_m:6
(assignvariableop_19_adam_dense_32_bias_m:R
@assignvariableop_20_adam_simple_rnn_2_simple_rnn_cell_2_kernel_m:@\
Jassignvariableop_21_adam_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_m:@@L
>assignvariableop_22_adam_simple_rnn_2_simple_rnn_cell_2_bias_m:@<
*assignvariableop_23_adam_dense_31_kernel_v:@6
(assignvariableop_24_adam_dense_31_bias_v:<
*assignvariableop_25_adam_dense_32_kernel_v:6
(assignvariableop_26_adam_dense_32_bias_v:R
@assignvariableop_27_adam_simple_rnn_2_simple_rnn_cell_2_kernel_v:@\
Jassignvariableop_28_adam_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_v:@@L
>assignvariableop_29_adam_simple_rnn_2_simple_rnn_cell_2_bias_v:@
identity_31ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9ќ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ф
valueкBзB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЃ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ї
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Р
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOpAssignVariableOp assignvariableop_dense_31_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_31_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_32_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_32_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_4AssignVariableOp8assignvariableop_4_simple_rnn_2_simple_rnn_cell_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_5AssignVariableOpBassignvariableop_5_simple_rnn_2_simple_rnn_cell_2_recurrent_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_6AssignVariableOp6assignvariableop_6_simple_rnn_2_simple_rnn_cell_2_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_iterIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_decayIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_11AssignVariableOp&assignvariableop_11_adam_learning_rateIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_31_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_dense_31_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_32_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_dense_32_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_20AssignVariableOp@assignvariableop_20_adam_simple_rnn_2_simple_rnn_cell_2_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_21AssignVariableOpJassignvariableop_21_adam_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_22AssignVariableOp>assignvariableop_22_adam_simple_rnn_2_simple_rnn_cell_2_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_31_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_31_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_32_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_32_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_simple_rnn_2_simple_rnn_cell_2_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_28AssignVariableOpJassignvariableop_28_adam_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_29AssignVariableOp>assignvariableop_29_adam_simple_rnn_2_simple_rnn_cell_2_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 г
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: –
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
»	
ц
E__inference_dense_32_layer_call_and_return_conditional_losses_1536515

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Л
“
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535606

inputs&
simple_rnn_2_1535588:@"
simple_rnn_2_1535590:@&
simple_rnn_2_1535592:@@"
dense_31_1535595:@
dense_31_1535597:"
dense_32_1535600:
dense_32_1535602:
identityИҐ dense_31/StatefulPartitionedCallҐ dense_32/StatefulPartitionedCallҐ$simple_rnn_2/StatefulPartitionedCallЫ
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_2_1535588simple_rnn_2_1535590simple_rnn_2_1535592*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1535555Ъ
 dense_31/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0dense_31_1535595dense_31_1535597*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_1535372Ц
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0dense_32_1535600dense_32_1535602*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_1535388x
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≥
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Е5
Я
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1535229

inputs+
simple_rnn_cell_2_1535152:@'
simple_rnn_cell_2_1535154:@+
simple_rnn_cell_2_1535156:@@
identityИҐ)simple_rnn_cell_2/StatefulPartitionedCallҐwhile;
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
valueB:—
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
:€€€€€€€€€@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskл
)simple_rnn_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_2_1535152simple_rnn_cell_2_1535154simple_rnn_cell_2_1535156*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_1535112n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :≈
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Т
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_2_1535152simple_rnn_cell_2_1535154simple_rnn_cell_2_1535156*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1535165*
condR
while_cond_1535164*8
output_shapes'
%: : : : :€€€€€€€€€@: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
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
:€€€€€€€€€@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@z
NoOpNoOp*^simple_rnn_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2V
)simple_rnn_cell_2/StatefulPartitionedCall)simple_rnn_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
я
ѓ
while_cond_1535286
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1535286___redundant_placeholder05
1while_while_cond_1535286___redundant_placeholder15
1while_while_cond_1535286___redundant_placeholder25
1while_while_cond_1535286___redundant_placeholder3
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
-: : : : :€€€€€€€€€@: ::::: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
«-
…
while_body_1536410
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:@G
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_2_matmul_readvariableop_resource:@E
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:@@ИҐ.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_2/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0¶
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0√
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@§
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Њ
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@™
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0™
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@ђ
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@w
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0 while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@я

while/NoOpNoOp/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€@: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
«-
…
while_body_1536080
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:@G
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_2_matmul_readvariableop_resource:@E
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:@@ИҐ.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_2/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0¶
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0√
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@§
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Њ
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@™
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0™
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@ђ
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@w
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0 while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@я

while/NoOpNoOp/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€@: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
ю
й
N__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_1534990

inputs

states0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates
£"
Ў
while_body_1535165
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
!while_simple_rnn_cell_2_1535187_0:@/
!while_simple_rnn_cell_2_1535189_0:@3
!while_simple_rnn_cell_2_1535191_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
while_simple_rnn_cell_2_1535187:@-
while_simple_rnn_cell_2_1535189:@1
while_simple_rnn_cell_2_1535191:@@ИҐ/while/simple_rnn_cell_2/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0¶
/while/simple_rnn_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_2_1535187_0!while_simple_rnn_cell_2_1535189_0!while_simple_rnn_cell_2_1535191_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_1535112r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Й
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:08while/simple_rnn_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
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
while/Identity_4Identity8while/simple_rnn_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@~

while/NoOpNoOp0^while/simple_rnn_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_2_1535187!while_simple_rnn_cell_2_1535187_0"D
while_simple_rnn_cell_2_1535189!while_simple_rnn_cell_2_1535189_0"D
while_simple_rnn_cell_2_1535191!while_simple_rnn_cell_2_1535191_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€@: : : : : 2b
/while/simple_rnn_cell_2/StatefulPartitionedCall/while/simple_rnn_cell_2/StatefulPartitionedCall: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
Ђ>
Љ
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1536367

inputsB
0simple_rnn_cell_2_matmul_readvariableop_resource:@?
1simple_rnn_cell_2_biasadd_readvariableop_resource:@D
2simple_rnn_cell_2_matmul_1_readvariableop_resource:@@
identityИҐ(simple_rnn_cell_2/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_2/MatMul/ReadVariableOpҐ)simple_rnn_cell_2/MatMul_1/ReadVariableOpҐwhile;
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
valueB:—
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
:€€€€€€€€€@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskШ
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Я
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ц
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ђ
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ь
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Щ
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ъ
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@k
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :≈
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1536300*
condR
while_cond_1536299*8
output_shapes'
%: : : : :€€€€€€€€€@: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
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
:€€€€€€€€€@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@ѕ
NoOpNoOp)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
яY
Д
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535871

inputsO
=simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resource:@L
>simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resource:@Q
?simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource:@@9
'dense_31_matmul_readvariableop_resource:@6
(dense_31_biasadd_readvariableop_resource:9
'dense_32_matmul_readvariableop_resource:6
(dense_32_biasadd_readvariableop_resource:
identityИҐdense_31/BiasAdd/ReadVariableOpҐdense_31/MatMul/ReadVariableOpҐdense_32/BiasAdd/ReadVariableOpҐdense_32/MatMul/ReadVariableOpҐ5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOpҐ4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOpҐ6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOpҐsimple_rnn_2/whileH
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
:€€€€€€€€€@p
simple_rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          З
simple_rnn_2/transpose	Transposeinputs$simple_rnn_2/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€^
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
€€€€€€€€€џ
simple_rnn_2/TensorArrayV2TensorListReserve1simple_rnn_2/TensorArrayV2/element_shape:output:0%simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“У
Bsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   З
4simple_rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_2/transpose:y:0Ksimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“l
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
valueB:™
simple_rnn_2/strided_slice_2StridedSlicesimple_rnn_2/transpose:y:0+simple_rnn_2/strided_slice_2/stack:output:0-simple_rnn_2/strided_slice_2/stack_1:output:0-simple_rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask≤
4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp=simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0∆
%simple_rnn_2/simple_rnn_cell_2/MatMulMatMul%simple_rnn_2/strided_slice_2:output:0<simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@∞
5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0”
&simple_rnn_2/simple_rnn_cell_2/BiasAddBiasAdd/simple_rnn_2/simple_rnn_cell_2/MatMul:product:0=simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@ґ
6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0ј
'simple_rnn_2/simple_rnn_cell_2/MatMul_1MatMulsimple_rnn_2/zeros:output:0>simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ѕ
"simple_rnn_2/simple_rnn_cell_2/addAddV2/simple_rnn_2/simple_rnn_cell_2/BiasAdd:output:01simple_rnn_2/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@Е
#simple_rnn_2/simple_rnn_cell_2/TanhTanh&simple_rnn_2/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@{
*simple_rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   k
)simple_rnn_2/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :м
simple_rnn_2/TensorArrayV2_1TensorListReserve3simple_rnn_2/TensorArrayV2_1/element_shape:output:02simple_rnn_2/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“S
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
€€€€€€€€€a
simple_rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Г
simple_rnn_2/whileWhile(simple_rnn_2/while/loop_counter:output:0.simple_rnn_2/while/maximum_iterations:output:0simple_rnn_2/time:output:0%simple_rnn_2/TensorArrayV2_1:handle:0simple_rnn_2/zeros:output:0%simple_rnn_2/strided_slice_1:output:0Dsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resource>simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resource?simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
simple_rnn_2_while_body_1535792*+
cond#R!
simple_rnn_2_while_cond_1535791*8
output_shapes'
%: : : : :€€€€€€€€€@: : : : : *
parallel_iterations О
=simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   э
/simple_rnn_2/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_2/while:output:3Fsimple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0*
num_elementsu
"simple_rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€n
$simple_rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
simple_rnn_2/strided_slice_3StridedSlice8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_2/strided_slice_3/stack:output:0-simple_rnn_2/strided_slice_3/stack_1:output:0-simple_rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_maskr
simple_rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          љ
simple_rnn_2/transpose_1	Transpose8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@Ж
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ъ
dense_31/MatMulMatMul%simple_rnn_2/strided_slice_3:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes

:*
dtype0О
dense_32/MatMulMatMuldense_31/BiasAdd:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€h
IdentityIdentitydense_32/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Й
NoOpNoOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp6^simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp5^simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp7^simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp^simple_rnn_2/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2n
5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp2l
4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp2p
6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp2(
simple_rnn_2/whilesimple_rnn_2/while:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
њ

¶
simple_rnn_2_while_cond_15359136
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_28
4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1O
Ksimple_rnn_2_while_simple_rnn_2_while_cond_1535913___redundant_placeholder0O
Ksimple_rnn_2_while_simple_rnn_2_while_cond_1535913___redundant_placeholder1O
Ksimple_rnn_2_while_simple_rnn_2_while_cond_1535913___redundant_placeholder2O
Ksimple_rnn_2_while_simple_rnn_2_while_cond_1535913___redundant_placeholder3
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
-: : : : :€€€€€€€€€@: ::::: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
њ

¶
simple_rnn_2_while_cond_15357916
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_28
4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1O
Ksimple_rnn_2_while_simple_rnn_2_while_cond_1535791___redundant_placeholder0O
Ksimple_rnn_2_while_simple_rnn_2_while_cond_1535791___redundant_placeholder1O
Ksimple_rnn_2_while_simple_rnn_2_while_cond_1535791___redundant_placeholder2O
Ksimple_rnn_2_while_simple_rnn_2_while_cond_1535791___redundant_placeholder3
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
-: : : : :€€€€€€€€€@: ::::: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
Ї
∞
-sequential_16_simple_rnn_2_while_cond_1534862R
Nsequential_16_simple_rnn_2_while_sequential_16_simple_rnn_2_while_loop_counterX
Tsequential_16_simple_rnn_2_while_sequential_16_simple_rnn_2_while_maximum_iterations0
,sequential_16_simple_rnn_2_while_placeholder2
.sequential_16_simple_rnn_2_while_placeholder_12
.sequential_16_simple_rnn_2_while_placeholder_2T
Psequential_16_simple_rnn_2_while_less_sequential_16_simple_rnn_2_strided_slice_1k
gsequential_16_simple_rnn_2_while_sequential_16_simple_rnn_2_while_cond_1534862___redundant_placeholder0k
gsequential_16_simple_rnn_2_while_sequential_16_simple_rnn_2_while_cond_1534862___redundant_placeholder1k
gsequential_16_simple_rnn_2_while_sequential_16_simple_rnn_2_while_cond_1534862___redundant_placeholder2k
gsequential_16_simple_rnn_2_while_sequential_16_simple_rnn_2_while_cond_1534862___redundant_placeholder3-
)sequential_16_simple_rnn_2_while_identity
ќ
%sequential_16/simple_rnn_2/while/LessLess,sequential_16_simple_rnn_2_while_placeholderPsequential_16_simple_rnn_2_while_less_sequential_16_simple_rnn_2_strided_slice_1*
T0*
_output_shapes
: Б
)sequential_16/simple_rnn_2/while/IdentityIdentity)sequential_16/simple_rnn_2/while/Less:z:0*
T0
*
_output_shapes
: "_
)sequential_16_simple_rnn_2_while_identity2sequential_16/simple_rnn_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€@: ::::: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
А
Є
.__inference_simple_rnn_2_layer_call_fn_1536026

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1535354o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ь	
І
%__inference_signature_wrapper_1535711
simple_rnn_2_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:
	unknown_4:
	unknown_5:
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__wrapped_model_1534942o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:€€€€€€€€€
,
_user_specified_namesimple_rnn_2_input
«-
…
while_body_1535287
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:@G
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_2_matmul_readvariableop_resource:@E
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:@@ИҐ.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_2/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0¶
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0√
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@§
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Њ
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@™
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0™
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@ђ
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@w
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0 while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype0:йи“M
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
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@я

while/NoOpNoOp/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€@: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
™	
•
/__inference_sequential_16_layer_call_fn_1535730

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:
	unknown_4:
	unknown_5:
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535395o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
А
Є
.__inference_simple_rnn_2_layer_call_fn_1536037

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1535555o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
»	
ц
E__inference_dense_31_layer_call_and_return_conditional_losses_1536496

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
я
ѓ
while_cond_1535164
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1535164___redundant_placeholder05
1while_while_cond_1535164___redundant_placeholder15
1while_while_cond_1535164___redundant_placeholder25
1while_while_cond_1535164___redundant_placeholder3
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
-: : : : :€€€€€€€€€@: ::::: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
ќ>
Њ
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1536257
inputs_0B
0simple_rnn_cell_2_matmul_readvariableop_resource:@?
1simple_rnn_cell_2_biasadd_readvariableop_resource:@D
2simple_rnn_cell_2_matmul_1_readvariableop_resource:@@
identityИҐ(simple_rnn_cell_2/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_2/MatMul/ReadVariableOpҐ)simple_rnn_cell_2/MatMul_1/ReadVariableOpҐwhile=
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
valueB:—
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
:€€€€€€€€€@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskШ
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Я
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ц
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ђ
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ь
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Щ
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ъ
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@k
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :≈
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1536190*
condR
while_cond_1536189*8
output_shapes'
%: : : : :€€€€€€€€€@: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
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
:€€€€€€€€€@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@ѕ
NoOpNoOp)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
ЕG
ч
-sequential_16_simple_rnn_2_while_body_1534863R
Nsequential_16_simple_rnn_2_while_sequential_16_simple_rnn_2_while_loop_counterX
Tsequential_16_simple_rnn_2_while_sequential_16_simple_rnn_2_while_maximum_iterations0
,sequential_16_simple_rnn_2_while_placeholder2
.sequential_16_simple_rnn_2_while_placeholder_12
.sequential_16_simple_rnn_2_while_placeholder_2Q
Msequential_16_simple_rnn_2_while_sequential_16_simple_rnn_2_strided_slice_1_0О
Йsequential_16_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0e
Ssequential_16_simple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0:@b
Tsequential_16_simple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:@g
Usequential_16_simple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:@@-
)sequential_16_simple_rnn_2_while_identity/
+sequential_16_simple_rnn_2_while_identity_1/
+sequential_16_simple_rnn_2_while_identity_2/
+sequential_16_simple_rnn_2_while_identity_3/
+sequential_16_simple_rnn_2_while_identity_4O
Ksequential_16_simple_rnn_2_while_sequential_16_simple_rnn_2_strided_slice_1М
Зsequential_16_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_2_tensorarrayunstack_tensorlistfromtensorc
Qsequential_16_simple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource:@`
Rsequential_16_simple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource:@e
Ssequential_16_simple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource:@@ИҐIsequential_16/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpҐHsequential_16/simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOpҐJsequential_16/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp£
Rsequential_16/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ѓ
Dsequential_16/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЙsequential_16_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0,sequential_16_simple_rnn_2_while_placeholder[sequential_16/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0№
Hsequential_16/simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpSsequential_16_simple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Ф
9sequential_16/simple_rnn_2/while/simple_rnn_cell_2/MatMulMatMulKsequential_16/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0Psequential_16/simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Џ
Isequential_16/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpTsequential_16_simple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0П
:sequential_16/simple_rnn_2/while/simple_rnn_cell_2/BiasAddBiasAddCsequential_16/simple_rnn_2/while/simple_rnn_cell_2/MatMul:product:0Qsequential_16/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@а
Jsequential_16/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpUsequential_16_simple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ы
;sequential_16/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1MatMul.sequential_16_simple_rnn_2_while_placeholder_2Rsequential_16/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@э
6sequential_16/simple_rnn_2/while/simple_rnn_cell_2/addAddV2Csequential_16/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd:output:0Esequential_16/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@≠
7sequential_16/simple_rnn_2/while/simple_rnn_cell_2/TanhTanh:sequential_16/simple_rnn_2/while/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Н
Ksequential_16/simple_rnn_2/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ё
Esequential_16/simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_16_simple_rnn_2_while_placeholder_1Tsequential_16/simple_rnn_2/while/TensorArrayV2Write/TensorListSetItem/index:output:0;sequential_16/simple_rnn_2/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype0:йи“h
&sequential_16/simple_rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :≠
$sequential_16/simple_rnn_2/while/addAddV2,sequential_16_simple_rnn_2_while_placeholder/sequential_16/simple_rnn_2/while/add/y:output:0*
T0*
_output_shapes
: j
(sequential_16/simple_rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :”
&sequential_16/simple_rnn_2/while/add_1AddV2Nsequential_16_simple_rnn_2_while_sequential_16_simple_rnn_2_while_loop_counter1sequential_16/simple_rnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: ™
)sequential_16/simple_rnn_2/while/IdentityIdentity*sequential_16/simple_rnn_2/while/add_1:z:0&^sequential_16/simple_rnn_2/while/NoOp*
T0*
_output_shapes
: ÷
+sequential_16/simple_rnn_2/while/Identity_1IdentityTsequential_16_simple_rnn_2_while_sequential_16_simple_rnn_2_while_maximum_iterations&^sequential_16/simple_rnn_2/while/NoOp*
T0*
_output_shapes
: ™
+sequential_16/simple_rnn_2/while/Identity_2Identity(sequential_16/simple_rnn_2/while/add:z:0&^sequential_16/simple_rnn_2/while/NoOp*
T0*
_output_shapes
: „
+sequential_16/simple_rnn_2/while/Identity_3IdentityUsequential_16/simple_rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^sequential_16/simple_rnn_2/while/NoOp*
T0*
_output_shapes
: ќ
+sequential_16/simple_rnn_2/while/Identity_4Identity;sequential_16/simple_rnn_2/while/simple_rnn_cell_2/Tanh:y:0&^sequential_16/simple_rnn_2/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Ћ
%sequential_16/simple_rnn_2/while/NoOpNoOpJ^sequential_16/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpI^sequential_16/simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOpK^sequential_16/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "_
)sequential_16_simple_rnn_2_while_identity2sequential_16/simple_rnn_2/while/Identity:output:0"c
+sequential_16_simple_rnn_2_while_identity_14sequential_16/simple_rnn_2/while/Identity_1:output:0"c
+sequential_16_simple_rnn_2_while_identity_24sequential_16/simple_rnn_2/while/Identity_2:output:0"c
+sequential_16_simple_rnn_2_while_identity_34sequential_16/simple_rnn_2/while/Identity_3:output:0"c
+sequential_16_simple_rnn_2_while_identity_44sequential_16/simple_rnn_2/while/Identity_4:output:0"Ь
Ksequential_16_simple_rnn_2_while_sequential_16_simple_rnn_2_strided_slice_1Msequential_16_simple_rnn_2_while_sequential_16_simple_rnn_2_strided_slice_1_0"™
Rsequential_16_simple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resourceTsequential_16_simple_rnn_2_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"ђ
Ssequential_16_simple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resourceUsequential_16_simple_rnn_2_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"®
Qsequential_16_simple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resourceSsequential_16_simple_rnn_2_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"Ц
Зsequential_16_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_2_tensorarrayunstack_tensorlistfromtensorЙsequential_16_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_16_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€@: : : : : 2Ц
Isequential_16/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpIsequential_16/simple_rnn_2/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2Ф
Hsequential_16/simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOpHsequential_16/simple_rnn_2/while/simple_rnn_cell_2/MatMul/ReadVariableOp2Ш
Jsequential_16/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpJsequential_16/simple_rnn_2/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
£"
Ў
while_body_1535004
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
!while_simple_rnn_cell_2_1535026_0:@/
!while_simple_rnn_cell_2_1535028_0:@3
!while_simple_rnn_cell_2_1535030_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
while_simple_rnn_cell_2_1535026:@-
while_simple_rnn_cell_2_1535028:@1
while_simple_rnn_cell_2_1535030:@@ИҐ/while/simple_rnn_cell_2/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0¶
/while/simple_rnn_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_2_1535026_0!while_simple_rnn_cell_2_1535028_0!while_simple_rnn_cell_2_1535030_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_1534990r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Й
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:08while/simple_rnn_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
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
while/Identity_4Identity8while/simple_rnn_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@~

while/NoOpNoOp0^while/simple_rnn_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_2_1535026!while_simple_rnn_cell_2_1535026_0"D
while_simple_rnn_cell_2_1535028!while_simple_rnn_cell_2_1535028_0"D
while_simple_rnn_cell_2_1535030!while_simple_rnn_cell_2_1535030_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€@: : : : : 2b
/while/simple_rnn_cell_2/StatefulPartitionedCall/while/simple_rnn_cell_2/StatefulPartitionedCall: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
ѓ
ё
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535684
simple_rnn_2_input&
simple_rnn_2_1535666:@"
simple_rnn_2_1535668:@&
simple_rnn_2_1535670:@@"
dense_31_1535673:@
dense_31_1535675:"
dense_32_1535678:
dense_32_1535680:
identityИҐ dense_31/StatefulPartitionedCallҐ dense_32/StatefulPartitionedCallҐ$simple_rnn_2/StatefulPartitionedCallІ
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputsimple_rnn_2_1535666simple_rnn_2_1535668simple_rnn_2_1535670*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1535555Ъ
 dense_31/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0dense_31_1535673dense_31_1535675*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_1535372Ц
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0dense_32_1535678dense_32_1535680*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_1535388x
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≥
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall:_ [
+
_output_shapes
:€€€€€€€€€
,
_user_specified_namesimple_rnn_2_input
яY
Д
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535993

inputsO
=simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resource:@L
>simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resource:@Q
?simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource:@@9
'dense_31_matmul_readvariableop_resource:@6
(dense_31_biasadd_readvariableop_resource:9
'dense_32_matmul_readvariableop_resource:6
(dense_32_biasadd_readvariableop_resource:
identityИҐdense_31/BiasAdd/ReadVariableOpҐdense_31/MatMul/ReadVariableOpҐdense_32/BiasAdd/ReadVariableOpҐdense_32/MatMul/ReadVariableOpҐ5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOpҐ4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOpҐ6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOpҐsimple_rnn_2/whileH
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
:€€€€€€€€€@p
simple_rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          З
simple_rnn_2/transpose	Transposeinputs$simple_rnn_2/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€^
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
€€€€€€€€€џ
simple_rnn_2/TensorArrayV2TensorListReserve1simple_rnn_2/TensorArrayV2/element_shape:output:0%simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“У
Bsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   З
4simple_rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_2/transpose:y:0Ksimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“l
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
valueB:™
simple_rnn_2/strided_slice_2StridedSlicesimple_rnn_2/transpose:y:0+simple_rnn_2/strided_slice_2/stack:output:0-simple_rnn_2/strided_slice_2/stack_1:output:0-simple_rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask≤
4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp=simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0∆
%simple_rnn_2/simple_rnn_cell_2/MatMulMatMul%simple_rnn_2/strided_slice_2:output:0<simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@∞
5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0”
&simple_rnn_2/simple_rnn_cell_2/BiasAddBiasAdd/simple_rnn_2/simple_rnn_cell_2/MatMul:product:0=simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@ґ
6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0ј
'simple_rnn_2/simple_rnn_cell_2/MatMul_1MatMulsimple_rnn_2/zeros:output:0>simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ѕ
"simple_rnn_2/simple_rnn_cell_2/addAddV2/simple_rnn_2/simple_rnn_cell_2/BiasAdd:output:01simple_rnn_2/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@Е
#simple_rnn_2/simple_rnn_cell_2/TanhTanh&simple_rnn_2/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@{
*simple_rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   k
)simple_rnn_2/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :м
simple_rnn_2/TensorArrayV2_1TensorListReserve3simple_rnn_2/TensorArrayV2_1/element_shape:output:02simple_rnn_2/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“S
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
€€€€€€€€€a
simple_rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Г
simple_rnn_2/whileWhile(simple_rnn_2/while/loop_counter:output:0.simple_rnn_2/while/maximum_iterations:output:0simple_rnn_2/time:output:0%simple_rnn_2/TensorArrayV2_1:handle:0simple_rnn_2/zeros:output:0%simple_rnn_2/strided_slice_1:output:0Dsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resource>simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resource?simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
simple_rnn_2_while_body_1535914*+
cond#R!
simple_rnn_2_while_cond_1535913*8
output_shapes'
%: : : : :€€€€€€€€€@: : : : : *
parallel_iterations О
=simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   э
/simple_rnn_2/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_2/while:output:3Fsimple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0*
num_elementsu
"simple_rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€n
$simple_rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
simple_rnn_2/strided_slice_3StridedSlice8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_2/strided_slice_3/stack:output:0-simple_rnn_2/strided_slice_3/stack_1:output:0-simple_rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_maskr
simple_rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          љ
simple_rnn_2/transpose_1	Transpose8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@Ж
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ъ
dense_31/MatMulMatMul%simple_rnn_2/strided_slice_3:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes

:*
dtype0О
dense_32/MatMulMatMuldense_31/BiasAdd:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€h
IdentityIdentitydense_32/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Й
NoOpNoOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp6^simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp5^simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp7^simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp^simple_rnn_2/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2n
5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp5simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp2l
4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp4simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp2p
6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp6simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp2(
simple_rnn_2/whilesimple_rnn_2/while:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Д
л
N__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_1536577

inputs
states_00
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0
Ђ>
Љ
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1535354

inputsB
0simple_rnn_cell_2_matmul_readvariableop_resource:@?
1simple_rnn_cell_2_biasadd_readvariableop_resource:@D
2simple_rnn_cell_2_matmul_1_readvariableop_resource:@@
identityИҐ(simple_rnn_cell_2/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_2/MatMul/ReadVariableOpҐ)simple_rnn_cell_2/MatMul_1/ReadVariableOpҐwhile;
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
valueB:—
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
:€€€€€€€€€@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
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
valueB:џ
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
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
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
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskШ
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Я
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ц
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ђ
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ь
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Щ
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ъ
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@k
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :≈
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
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
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1535287*
condR
while_cond_1535286*8
output_shapes'
%: : : : :€€€€€€€€€@: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
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
:€€€€€€€€€@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@ѕ
NoOpNoOp)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
я
ѓ
while_cond_1536189
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1536189___redundant_placeholder05
1while_while_cond_1536189___redundant_placeholder15
1while_while_cond_1536189___redundant_placeholder25
1while_while_cond_1536189___redundant_placeholder3
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
-: : : : :€€€€€€€€€@: ::::: 
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
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
ƒ
Ч
*__inference_dense_31_layer_call_fn_1536486

inputs
unknown:@
	unknown_0:
identityИҐStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_1535372o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Л
“
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535395

inputs&
simple_rnn_2_1535355:@"
simple_rnn_2_1535357:@&
simple_rnn_2_1535359:@@"
dense_31_1535373:@
dense_31_1535375:"
dense_32_1535389:
dense_32_1535391:
identityИҐ dense_31/StatefulPartitionedCallҐ dense_32/StatefulPartitionedCallҐ$simple_rnn_2/StatefulPartitionedCallЫ
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_2_1535355simple_rnn_2_1535357simple_rnn_2_1535359*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1535354Ъ
 dense_31/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0dense_31_1535373dense_31_1535375*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_1535372Ц
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0dense_32_1535389dense_32_1535391*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_1535388x
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≥
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ўk
Ї
"__inference__wrapped_model_1534942
simple_rnn_2_input]
Ksequential_16_simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resource:@Z
Lsequential_16_simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resource:@_
Msequential_16_simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource:@@G
5sequential_16_dense_31_matmul_readvariableop_resource:@D
6sequential_16_dense_31_biasadd_readvariableop_resource:G
5sequential_16_dense_32_matmul_readvariableop_resource:D
6sequential_16_dense_32_biasadd_readvariableop_resource:
identityИҐ-sequential_16/dense_31/BiasAdd/ReadVariableOpҐ,sequential_16/dense_31/MatMul/ReadVariableOpҐ-sequential_16/dense_32/BiasAdd/ReadVariableOpҐ,sequential_16/dense_32/MatMul/ReadVariableOpҐCsequential_16/simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOpҐBsequential_16/simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOpҐDsequential_16/simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOpҐ sequential_16/simple_rnn_2/whileb
 sequential_16/simple_rnn_2/ShapeShapesimple_rnn_2_input*
T0*
_output_shapes
:x
.sequential_16/simple_rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential_16/simple_rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential_16/simple_rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
(sequential_16/simple_rnn_2/strided_sliceStridedSlice)sequential_16/simple_rnn_2/Shape:output:07sequential_16/simple_rnn_2/strided_slice/stack:output:09sequential_16/simple_rnn_2/strided_slice/stack_1:output:09sequential_16/simple_rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_16/simple_rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@ƒ
'sequential_16/simple_rnn_2/zeros/packedPack1sequential_16/simple_rnn_2/strided_slice:output:02sequential_16/simple_rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&sequential_16/simple_rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    љ
 sequential_16/simple_rnn_2/zerosFill0sequential_16/simple_rnn_2/zeros/packed:output:0/sequential_16/simple_rnn_2/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@~
)sequential_16/simple_rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ѓ
$sequential_16/simple_rnn_2/transpose	Transposesimple_rnn_2_input2sequential_16/simple_rnn_2/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€z
"sequential_16/simple_rnn_2/Shape_1Shape(sequential_16/simple_rnn_2/transpose:y:0*
T0*
_output_shapes
:z
0sequential_16/simple_rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_16/simple_rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_16/simple_rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
*sequential_16/simple_rnn_2/strided_slice_1StridedSlice+sequential_16/simple_rnn_2/Shape_1:output:09sequential_16/simple_rnn_2/strided_slice_1/stack:output:0;sequential_16/simple_rnn_2/strided_slice_1/stack_1:output:0;sequential_16/simple_rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
6sequential_16/simple_rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Е
(sequential_16/simple_rnn_2/TensorArrayV2TensorListReserve?sequential_16/simple_rnn_2/TensorArrayV2/element_shape:output:03sequential_16/simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“°
Psequential_16/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ±
Bsequential_16/simple_rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_16/simple_rnn_2/transpose:y:0Ysequential_16/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“z
0sequential_16/simple_rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_16/simple_rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_16/simple_rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
*sequential_16/simple_rnn_2/strided_slice_2StridedSlice(sequential_16/simple_rnn_2/transpose:y:09sequential_16/simple_rnn_2/strided_slice_2/stack:output:0;sequential_16/simple_rnn_2/strided_slice_2/stack_1:output:0;sequential_16/simple_rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskќ
Bsequential_16/simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpKsequential_16_simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0р
3sequential_16/simple_rnn_2/simple_rnn_cell_2/MatMulMatMul3sequential_16/simple_rnn_2/strided_slice_2:output:0Jsequential_16/simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@ћ
Csequential_16/simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpLsequential_16_simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0э
4sequential_16/simple_rnn_2/simple_rnn_cell_2/BiasAddBiasAdd=sequential_16/simple_rnn_2/simple_rnn_cell_2/MatMul:product:0Ksequential_16/simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@“
Dsequential_16/simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpMsequential_16_simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0к
5sequential_16/simple_rnn_2/simple_rnn_cell_2/MatMul_1MatMul)sequential_16/simple_rnn_2/zeros:output:0Lsequential_16/simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@л
0sequential_16/simple_rnn_2/simple_rnn_cell_2/addAddV2=sequential_16/simple_rnn_2/simple_rnn_cell_2/BiasAdd:output:0?sequential_16/simple_rnn_2/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@°
1sequential_16/simple_rnn_2/simple_rnn_cell_2/TanhTanh4sequential_16/simple_rnn_2/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Й
8sequential_16/simple_rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   y
7sequential_16/simple_rnn_2/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ц
*sequential_16/simple_rnn_2/TensorArrayV2_1TensorListReserveAsequential_16/simple_rnn_2/TensorArrayV2_1/element_shape:output:0@sequential_16/simple_rnn_2/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“a
sequential_16/simple_rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3sequential_16/simple_rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€o
-sequential_16/simple_rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : є
 sequential_16/simple_rnn_2/whileWhile6sequential_16/simple_rnn_2/while/loop_counter:output:0<sequential_16/simple_rnn_2/while/maximum_iterations:output:0(sequential_16/simple_rnn_2/time:output:03sequential_16/simple_rnn_2/TensorArrayV2_1:handle:0)sequential_16/simple_rnn_2/zeros:output:03sequential_16/simple_rnn_2/strided_slice_1:output:0Rsequential_16/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ksequential_16_simple_rnn_2_simple_rnn_cell_2_matmul_readvariableop_resourceLsequential_16_simple_rnn_2_simple_rnn_cell_2_biasadd_readvariableop_resourceMsequential_16_simple_rnn_2_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *9
body1R/
-sequential_16_simple_rnn_2_while_body_1534863*9
cond1R/
-sequential_16_simple_rnn_2_while_cond_1534862*8
output_shapes'
%: : : : :€€€€€€€€€@: : : : : *
parallel_iterations Ь
Ksequential_16/simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   І
=sequential_16/simple_rnn_2/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_16/simple_rnn_2/while:output:3Tsequential_16/simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0*
num_elementsГ
0sequential_16/simple_rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€|
2sequential_16/simple_rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2sequential_16/simple_rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
*sequential_16/simple_rnn_2/strided_slice_3StridedSliceFsequential_16/simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:09sequential_16/simple_rnn_2/strided_slice_3/stack:output:0;sequential_16/simple_rnn_2/strided_slice_3/stack_1:output:0;sequential_16/simple_rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_maskА
+sequential_16/simple_rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          з
&sequential_16/simple_rnn_2/transpose_1	TransposeFsequential_16/simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:04sequential_16/simple_rnn_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@Ґ
,sequential_16/dense_31/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_31_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0ƒ
sequential_16/dense_31/MatMulMatMul3sequential_16/simple_rnn_2/strided_slice_3:output:04sequential_16/dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
-sequential_16/dense_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ї
sequential_16/dense_31/BiasAddBiasAdd'sequential_16/dense_31/MatMul:product:05sequential_16/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ґ
,sequential_16/dense_32/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_32_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Є
sequential_16/dense_32/MatMulMatMul'sequential_16/dense_31/BiasAdd:output:04sequential_16/dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
-sequential_16/dense_32/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ї
sequential_16/dense_32/BiasAddBiasAdd'sequential_16/dense_32/MatMul:product:05sequential_16/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€v
IdentityIdentity'sequential_16/dense_32/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€щ
NoOpNoOp.^sequential_16/dense_31/BiasAdd/ReadVariableOp-^sequential_16/dense_31/MatMul/ReadVariableOp.^sequential_16/dense_32/BiasAdd/ReadVariableOp-^sequential_16/dense_32/MatMul/ReadVariableOpD^sequential_16/simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOpC^sequential_16/simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOpE^sequential_16/simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp!^sequential_16/simple_rnn_2/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 2^
-sequential_16/dense_31/BiasAdd/ReadVariableOp-sequential_16/dense_31/BiasAdd/ReadVariableOp2\
,sequential_16/dense_31/MatMul/ReadVariableOp,sequential_16/dense_31/MatMul/ReadVariableOp2^
-sequential_16/dense_32/BiasAdd/ReadVariableOp-sequential_16/dense_32/BiasAdd/ReadVariableOp2\
,sequential_16/dense_32/MatMul/ReadVariableOp,sequential_16/dense_32/MatMul/ReadVariableOp2К
Csequential_16/simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOpCsequential_16/simple_rnn_2/simple_rnn_cell_2/BiasAdd/ReadVariableOp2И
Bsequential_16/simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOpBsequential_16/simple_rnn_2/simple_rnn_cell_2/MatMul/ReadVariableOp2М
Dsequential_16/simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOpDsequential_16/simple_rnn_2/simple_rnn_cell_2/MatMul_1/ReadVariableOp2D
 sequential_16/simple_rnn_2/while sequential_16/simple_rnn_2/while:_ [
+
_output_shapes
:€€€€€€€€€
,
_user_specified_namesimple_rnn_2_input
ќ	
±
/__inference_sequential_16_layer_call_fn_1535642
simple_rnn_2_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:
	unknown_4:
	unknown_5:
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535606o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:€€€€€€€€€
,
_user_specified_namesimple_rnn_2_input
Д
л
N__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_1536560

inputs
states_00
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*≈
serving_default±
U
simple_rnn_2_input?
$serving_default_simple_rnn_2_input:0€€€€€€€€€<
dense_320
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:Л≠
џ
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
√
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
ї
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ї
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
Q
%0
&1
'2
3
4
#5
$6"
trackable_list_wrapper
Q
%0
&1
'2
3
4
#5
$6"
trackable_list_wrapper
 "
trackable_list_wrapper
 
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
с
-trace_0
.trace_1
/trace_2
0trace_32Ж
/__inference_sequential_16_layer_call_fn_1535412
/__inference_sequential_16_layer_call_fn_1535730
/__inference_sequential_16_layer_call_fn_1535749
/__inference_sequential_16_layer_call_fn_1535642њ
ґ≤≤
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
annotations™ *
 z-trace_0z.trace_1z/trace_2z0trace_3
Ё
1trace_0
2trace_1
3trace_2
4trace_32т
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535871
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535993
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535663
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535684њ
ґ≤≤
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
annotations™ *
 z1trace_0z2trace_1z3trace_2z4trace_3
ЎB’
"__inference__wrapped_model_1534942simple_rnn_2_input"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—
5iter

6beta_1

7beta_2
	8decay
9learning_ratemqmr#ms$mt%mu&mv'mwvxvy#vz$v{%v|&v}'v~"
	optimizer
,
:serving_default"
signature_map
5
%0
&1
'2"
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
є

;states
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
В
Atrace_0
Btrace_1
Ctrace_2
Dtrace_32Ч
.__inference_simple_rnn_2_layer_call_fn_1536004
.__inference_simple_rnn_2_layer_call_fn_1536015
.__inference_simple_rnn_2_layer_call_fn_1536026
.__inference_simple_rnn_2_layer_call_fn_1536037‘
Ћ≤«
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
annotations™ *
 zAtrace_0zBtrace_1zCtrace_2zDtrace_3
о
Etrace_0
Ftrace_1
Gtrace_2
Htrace_32Г
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1536147
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1536257
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1536367
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1536477‘
Ћ≤«
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
annotations™ *
 zEtrace_0zFtrace_1zGtrace_2zHtrace_3
и
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
O_random_generator

%kernel
&recurrent_kernel
'bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
о
Utrace_02—
*__inference_dense_31_layer_call_fn_1536486Ґ
Щ≤Х
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
annotations™ *
 zUtrace_0
Й
Vtrace_02м
E__inference_dense_31_layer_call_and_return_conditional_losses_1536496Ґ
Щ≤Х
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
annotations™ *
 zVtrace_0
!:@2dense_31/kernel
:2dense_31/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
о
\trace_02—
*__inference_dense_32_layer_call_fn_1536505Ґ
Щ≤Х
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
annotations™ *
 z\trace_0
Й
]trace_02м
E__inference_dense_32_layer_call_and_return_conditional_losses_1536515Ґ
Щ≤Х
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
annotations™ *
 z]trace_0
!:2dense_32/kernel
:2dense_32/bias
7:5@2%simple_rnn_2/simple_rnn_cell_2/kernel
A:?@@2/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel
1:/@2#simple_rnn_2/simple_rnn_cell_2/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
МBЙ
/__inference_sequential_16_layer_call_fn_1535412simple_rnn_2_input"њ
ґ≤≤
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
annotations™ *
 
АBэ
/__inference_sequential_16_layer_call_fn_1535730inputs"њ
ґ≤≤
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
annotations™ *
 
АBэ
/__inference_sequential_16_layer_call_fn_1535749inputs"њ
ґ≤≤
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
annotations™ *
 
МBЙ
/__inference_sequential_16_layer_call_fn_1535642simple_rnn_2_input"њ
ґ≤≤
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
annotations™ *
 
ЫBШ
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535871inputs"њ
ґ≤≤
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
annotations™ *
 
ЫBШ
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535993inputs"њ
ґ≤≤
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
annotations™ *
 
ІB§
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535663simple_rnn_2_input"њ
ґ≤≤
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
annotations™ *
 
ІB§
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535684simple_rnn_2_input"њ
ґ≤≤
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
annotations™ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
„B‘
%__inference_signature_wrapper_1535711simple_rnn_2_input"Ф
Н≤Й
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
annotations™ *
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
.__inference_simple_rnn_2_layer_call_fn_1536004inputs/0"‘
Ћ≤«
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
annotations™ *
 
ЦBУ
.__inference_simple_rnn_2_layer_call_fn_1536015inputs/0"‘
Ћ≤«
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
annotations™ *
 
ФBС
.__inference_simple_rnn_2_layer_call_fn_1536026inputs"‘
Ћ≤«
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
annotations™ *
 
ФBС
.__inference_simple_rnn_2_layer_call_fn_1536037inputs"‘
Ћ≤«
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
annotations™ *
 
±BЃ
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1536147inputs/0"‘
Ћ≤«
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
annotations™ *
 
±BЃ
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1536257inputs/0"‘
Ћ≤«
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
annotations™ *
 
ѓBђ
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1536367inputs"‘
Ћ≤«
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
annotations™ *
 
ѓBђ
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1536477inputs"‘
Ћ≤«
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
annotations™ *
 
5
%0
&1
'2"
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
б
etrace_0
ftrace_12™
3__inference_simple_rnn_cell_2_layer_call_fn_1536529
3__inference_simple_rnn_cell_2_layer_call_fn_1536543љ
і≤∞
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
annotations™ *
 zetrace_0zftrace_1
Ч
gtrace_0
htrace_12а
N__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_1536560
N__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_1536577љ
і≤∞
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
annotations™ *
 zgtrace_0zhtrace_1
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
ёBџ
*__inference_dense_31_layer_call_fn_1536486inputs"Ґ
Щ≤Х
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
annotations™ *
 
щBц
E__inference_dense_31_layer_call_and_return_conditional_losses_1536496inputs"Ґ
Щ≤Х
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
annotations™ *
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
ёBџ
*__inference_dense_32_layer_call_fn_1536505inputs"Ґ
Щ≤Х
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
annotations™ *
 
щBц
E__inference_dense_32_layer_call_and_return_conditional_losses_1536515inputs"Ґ
Щ≤Х
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
annotations™ *
 
N
i	variables
j	keras_api
	ktotal
	lcount"
_tf_keras_metric
N
m	variables
n	keras_api
	ototal
	pcount"
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
3__inference_simple_rnn_cell_2_layer_call_fn_1536529inputsstates/0"љ
і≤∞
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
annotations™ *
 
МBЙ
3__inference_simple_rnn_cell_2_layer_call_fn_1536543inputsstates/0"љ
і≤∞
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
annotations™ *
 
ІB§
N__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_1536560inputsstates/0"љ
і≤∞
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
annotations™ *
 
ІB§
N__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_1536577inputsstates/0"љ
і≤∞
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
annotations™ *
 
.
k0
l1"
trackable_list_wrapper
-
i	variables"
_generic_user_object
:  (2total
:  (2count
.
o0
p1"
trackable_list_wrapper
-
m	variables"
_generic_user_object
:  (2total
:  (2count
&:$@2Adam/dense_31/kernel/m
 :2Adam/dense_31/bias/m
&:$2Adam/dense_32/kernel/m
 :2Adam/dense_32/bias/m
<::@2,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/m
F:D@@26Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/m
6:4@2*Adam/simple_rnn_2/simple_rnn_cell_2/bias/m
&:$@2Adam/dense_31/kernel/v
 :2Adam/dense_31/bias/v
&:$2Adam/dense_32/kernel/v
 :2Adam/dense_32/bias/v
<::@2,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/v
F:D@@26Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/v
6:4@2*Adam/simple_rnn_2/simple_rnn_cell_2/bias/v•
"__inference__wrapped_model_1534942%'&#$?Ґ<
5Ґ2
0К-
simple_rnn_2_input€€€€€€€€€
™ "3™0
.
dense_32"К
dense_32€€€€€€€€€•
E__inference_dense_31_layer_call_and_return_conditional_losses_1536496\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€
Ъ }
*__inference_dense_31_layer_call_fn_1536486O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€•
E__inference_dense_32_layer_call_and_return_conditional_losses_1536515\#$/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ }
*__inference_dense_32_layer_call_fn_1536505O#$/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€«
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535663y%'&#$GҐD
=Ґ:
0К-
simple_rnn_2_input€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ «
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535684y%'&#$GҐD
=Ґ:
0К-
simple_rnn_2_input€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ї
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535871m%'&#$;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ї
J__inference_sequential_16_layer_call_and_return_conditional_losses_1535993m%'&#$;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Я
/__inference_sequential_16_layer_call_fn_1535412l%'&#$GҐD
=Ґ:
0К-
simple_rnn_2_input€€€€€€€€€
p 

 
™ "К€€€€€€€€€Я
/__inference_sequential_16_layer_call_fn_1535642l%'&#$GҐD
=Ґ:
0К-
simple_rnn_2_input€€€€€€€€€
p

 
™ "К€€€€€€€€€У
/__inference_sequential_16_layer_call_fn_1535730`%'&#$;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€У
/__inference_sequential_16_layer_call_fn_1535749`%'&#$;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€њ
%__inference_signature_wrapper_1535711Х%'&#$UҐR
Ґ 
K™H
F
simple_rnn_2_input0К-
simple_rnn_2_input€€€€€€€€€"3™0
.
dense_32"К
dense_32€€€€€€€€€ 
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1536147}%'&OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ  
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1536257}%'&OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Ї
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1536367m%'&?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Ї
I__inference_simple_rnn_2_layer_call_and_return_conditional_losses_1536477m%'&?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Ґ
.__inference_simple_rnn_2_layer_call_fn_1536004p%'&OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€@Ґ
.__inference_simple_rnn_2_layer_call_fn_1536015p%'&OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "К€€€€€€€€€@Т
.__inference_simple_rnn_2_layer_call_fn_1536026`%'&?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€@Т
.__inference_simple_rnn_2_layer_call_fn_1536037`%'&?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ "К€€€€€€€€€@К
N__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_1536560Ј%'&\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states/0€€€€€€€€€@
p 
™ "RҐO
HҐE
К
0/0€€€€€€€€€@
$Ъ!
К
0/1/0€€€€€€€€€@
Ъ К
N__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_1536577Ј%'&\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states/0€€€€€€€€€@
p
™ "RҐO
HҐE
К
0/0€€€€€€€€€@
$Ъ!
К
0/1/0€€€€€€€€€@
Ъ б
3__inference_simple_rnn_cell_2_layer_call_fn_1536529©%'&\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states/0€€€€€€€€€@
p 
™ "DҐA
К
0€€€€€€€€€@
"Ъ
К
1/0€€€€€€€€€@б
3__inference_simple_rnn_cell_2_layer_call_fn_1536543©%'&\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states/0€€€€€€€€€@
p
™ "DҐA
К
0€€€€€€€€€@
"Ъ
К
1/0€€€€€€€€€@