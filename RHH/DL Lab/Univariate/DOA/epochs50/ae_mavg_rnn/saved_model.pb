«ф8
 Ъ
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
И"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ЄЬ5
Ц
Adam/time_distributed_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/time_distributed_18/bias/v
П
3Adam/time_distributed_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/time_distributed_18/bias/v*
_output_shapes
:*
dtype0
Ю
!Adam/time_distributed_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!Adam/time_distributed_18/kernel/v
Ч
5Adam/time_distributed_18/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/time_distributed_18/kernel/v*
_output_shapes

: *
dtype0
∞
,Adam/simple_rnn_11/simple_rnn_cell_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/simple_rnn_11/simple_rnn_cell_11/bias/v
©
@Adam/simple_rnn_11/simple_rnn_cell_11/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_11/simple_rnn_cell_11/bias/v*
_output_shapes
: *
dtype0
ћ
8Adam/simple_rnn_11/simple_rnn_cell_11/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *I
shared_name:8Adam/simple_rnn_11/simple_rnn_cell_11/recurrent_kernel/v
≈
LAdam/simple_rnn_11/simple_rnn_cell_11/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_11/simple_rnn_cell_11/recurrent_kernel/v*
_output_shapes

:  *
dtype0
Є
.Adam/simple_rnn_11/simple_rnn_cell_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *?
shared_name0.Adam/simple_rnn_11/simple_rnn_cell_11/kernel/v
±
BAdam/simple_rnn_11/simple_rnn_cell_11/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_11/simple_rnn_cell_11/kernel/v*
_output_shapes

: *
dtype0
∞
,Adam/simple_rnn_10/simple_rnn_cell_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/simple_rnn_10/simple_rnn_cell_10/bias/v
©
@Adam/simple_rnn_10/simple_rnn_cell_10/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_10/simple_rnn_cell_10/bias/v*
_output_shapes
:*
dtype0
ћ
8Adam/simple_rnn_10/simple_rnn_cell_10/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*I
shared_name:8Adam/simple_rnn_10/simple_rnn_cell_10/recurrent_kernel/v
≈
LAdam/simple_rnn_10/simple_rnn_cell_10/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_10/simple_rnn_cell_10/recurrent_kernel/v*
_output_shapes

:*
dtype0
Є
.Adam/simple_rnn_10/simple_rnn_cell_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*?
shared_name0.Adam/simple_rnn_10/simple_rnn_cell_10/kernel/v
±
BAdam/simple_rnn_10/simple_rnn_cell_10/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_10/simple_rnn_cell_10/kernel/v*
_output_shapes

:*
dtype0
ђ
*Adam/simple_rnn_9/simple_rnn_cell_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/simple_rnn_9/simple_rnn_cell_9/bias/v
•
>Adam/simple_rnn_9/simple_rnn_cell_9/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_9/simple_rnn_cell_9/bias/v*
_output_shapes
:*
dtype0
»
6Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/v
Ѕ
JAdam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/v*
_output_shapes

:*
dtype0
і
,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *=
shared_name.,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/v
≠
@Adam/simple_rnn_9/simple_rnn_cell_9/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/v*
_output_shapes

: *
dtype0
ђ
*Adam/simple_rnn_8/simple_rnn_cell_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/simple_rnn_8/simple_rnn_cell_8/bias/v
•
>Adam/simple_rnn_8/simple_rnn_cell_8/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_8/simple_rnn_cell_8/bias/v*
_output_shapes
: *
dtype0
»
6Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *G
shared_name86Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/v
Ѕ
JAdam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/v*
_output_shapes

:  *
dtype0
і
,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *=
shared_name.,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/v
≠
@Adam/simple_rnn_8/simple_rnn_cell_8/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/v*
_output_shapes

: *
dtype0
Ц
Adam/time_distributed_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/time_distributed_18/bias/m
П
3Adam/time_distributed_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/time_distributed_18/bias/m*
_output_shapes
:*
dtype0
Ю
!Adam/time_distributed_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!Adam/time_distributed_18/kernel/m
Ч
5Adam/time_distributed_18/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/time_distributed_18/kernel/m*
_output_shapes

: *
dtype0
∞
,Adam/simple_rnn_11/simple_rnn_cell_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/simple_rnn_11/simple_rnn_cell_11/bias/m
©
@Adam/simple_rnn_11/simple_rnn_cell_11/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_11/simple_rnn_cell_11/bias/m*
_output_shapes
: *
dtype0
ћ
8Adam/simple_rnn_11/simple_rnn_cell_11/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *I
shared_name:8Adam/simple_rnn_11/simple_rnn_cell_11/recurrent_kernel/m
≈
LAdam/simple_rnn_11/simple_rnn_cell_11/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_11/simple_rnn_cell_11/recurrent_kernel/m*
_output_shapes

:  *
dtype0
Є
.Adam/simple_rnn_11/simple_rnn_cell_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *?
shared_name0.Adam/simple_rnn_11/simple_rnn_cell_11/kernel/m
±
BAdam/simple_rnn_11/simple_rnn_cell_11/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_11/simple_rnn_cell_11/kernel/m*
_output_shapes

: *
dtype0
∞
,Adam/simple_rnn_10/simple_rnn_cell_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/simple_rnn_10/simple_rnn_cell_10/bias/m
©
@Adam/simple_rnn_10/simple_rnn_cell_10/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_10/simple_rnn_cell_10/bias/m*
_output_shapes
:*
dtype0
ћ
8Adam/simple_rnn_10/simple_rnn_cell_10/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*I
shared_name:8Adam/simple_rnn_10/simple_rnn_cell_10/recurrent_kernel/m
≈
LAdam/simple_rnn_10/simple_rnn_cell_10/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_10/simple_rnn_cell_10/recurrent_kernel/m*
_output_shapes

:*
dtype0
Є
.Adam/simple_rnn_10/simple_rnn_cell_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*?
shared_name0.Adam/simple_rnn_10/simple_rnn_cell_10/kernel/m
±
BAdam/simple_rnn_10/simple_rnn_cell_10/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_10/simple_rnn_cell_10/kernel/m*
_output_shapes

:*
dtype0
ђ
*Adam/simple_rnn_9/simple_rnn_cell_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/simple_rnn_9/simple_rnn_cell_9/bias/m
•
>Adam/simple_rnn_9/simple_rnn_cell_9/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_9/simple_rnn_cell_9/bias/m*
_output_shapes
:*
dtype0
»
6Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/m
Ѕ
JAdam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/m*
_output_shapes

:*
dtype0
і
,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *=
shared_name.,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/m
≠
@Adam/simple_rnn_9/simple_rnn_cell_9/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/m*
_output_shapes

: *
dtype0
ђ
*Adam/simple_rnn_8/simple_rnn_cell_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/simple_rnn_8/simple_rnn_cell_8/bias/m
•
>Adam/simple_rnn_8/simple_rnn_cell_8/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_8/simple_rnn_cell_8/bias/m*
_output_shapes
: *
dtype0
»
6Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *G
shared_name86Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/m
Ѕ
JAdam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/m*
_output_shapes

:  *
dtype0
і
,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *=
shared_name.,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/m
≠
@Adam/simple_rnn_8/simple_rnn_cell_8/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/m*
_output_shapes

: *
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
И
time_distributed_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nametime_distributed_18/bias
Б
,time_distributed_18/bias/Read/ReadVariableOpReadVariableOptime_distributed_18/bias*
_output_shapes
:*
dtype0
Р
time_distributed_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *+
shared_nametime_distributed_18/kernel
Й
.time_distributed_18/kernel/Read/ReadVariableOpReadVariableOptime_distributed_18/kernel*
_output_shapes

: *
dtype0
Ґ
%simple_rnn_11/simple_rnn_cell_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%simple_rnn_11/simple_rnn_cell_11/bias
Ы
9simple_rnn_11/simple_rnn_cell_11/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_11/simple_rnn_cell_11/bias*
_output_shapes
: *
dtype0
Њ
1simple_rnn_11/simple_rnn_cell_11/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *B
shared_name31simple_rnn_11/simple_rnn_cell_11/recurrent_kernel
Ј
Esimple_rnn_11/simple_rnn_cell_11/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_11/simple_rnn_cell_11/recurrent_kernel*
_output_shapes

:  *
dtype0
™
'simple_rnn_11/simple_rnn_cell_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *8
shared_name)'simple_rnn_11/simple_rnn_cell_11/kernel
£
;simple_rnn_11/simple_rnn_cell_11/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_11/simple_rnn_cell_11/kernel*
_output_shapes

: *
dtype0
Ґ
%simple_rnn_10/simple_rnn_cell_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%simple_rnn_10/simple_rnn_cell_10/bias
Ы
9simple_rnn_10/simple_rnn_cell_10/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_10/simple_rnn_cell_10/bias*
_output_shapes
:*
dtype0
Њ
1simple_rnn_10/simple_rnn_cell_10/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31simple_rnn_10/simple_rnn_cell_10/recurrent_kernel
Ј
Esimple_rnn_10/simple_rnn_cell_10/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_10/simple_rnn_cell_10/recurrent_kernel*
_output_shapes

:*
dtype0
™
'simple_rnn_10/simple_rnn_cell_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'simple_rnn_10/simple_rnn_cell_10/kernel
£
;simple_rnn_10/simple_rnn_cell_10/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_10/simple_rnn_cell_10/kernel*
_output_shapes

:*
dtype0
Ю
#simple_rnn_9/simple_rnn_cell_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#simple_rnn_9/simple_rnn_cell_9/bias
Ч
7simple_rnn_9/simple_rnn_cell_9/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_9/simple_rnn_cell_9/bias*
_output_shapes
:*
dtype0
Ї
/simple_rnn_9/simple_rnn_cell_9/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@
shared_name1/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel
≥
Csimple_rnn_9/simple_rnn_cell_9/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel*
_output_shapes

:*
dtype0
¶
%simple_rnn_9/simple_rnn_cell_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *6
shared_name'%simple_rnn_9/simple_rnn_cell_9/kernel
Я
9simple_rnn_9/simple_rnn_cell_9/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_9/simple_rnn_cell_9/kernel*
_output_shapes

: *
dtype0
Ю
#simple_rnn_8/simple_rnn_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#simple_rnn_8/simple_rnn_cell_8/bias
Ч
7simple_rnn_8/simple_rnn_cell_8/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_8/simple_rnn_cell_8/bias*
_output_shapes
: *
dtype0
Ї
/simple_rnn_8/simple_rnn_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *@
shared_name1/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel
≥
Csimple_rnn_8/simple_rnn_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel*
_output_shapes

:  *
dtype0
¶
%simple_rnn_8/simple_rnn_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *6
shared_name'%simple_rnn_8/simple_rnn_cell_8/kernel
Я
9simple_rnn_8/simple_rnn_cell_8/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_8/simple_rnn_cell_8/kernel*
_output_shapes

: *
dtype0
Н
"serving_default_simple_rnn_8_inputPlaceholder*+
_output_shapes
:€€€€€€€€€*
dtype0* 
shape:€€€€€€€€€
Ф
StatefulPartitionedCallStatefulPartitionedCall"serving_default_simple_rnn_8_input%simple_rnn_8/simple_rnn_cell_8/kernel#simple_rnn_8/simple_rnn_cell_8/bias/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel%simple_rnn_9/simple_rnn_cell_9/kernel#simple_rnn_9/simple_rnn_cell_9/bias/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel'simple_rnn_10/simple_rnn_cell_10/kernel%simple_rnn_10/simple_rnn_cell_10/bias1simple_rnn_10/simple_rnn_cell_10/recurrent_kernel'simple_rnn_11/simple_rnn_cell_11/kernel%simple_rnn_11/simple_rnn_cell_11/bias1simple_rnn_11/simple_rnn_cell_11/recurrent_kerneltime_distributed_18/kerneltime_distributed_18/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_1737291

NoOpNoOp
’t
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Рt
valueЖtBГt Bьs
Ь
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
™
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec*
™
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec*
О
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses* 
™
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,cell
-
state_spec*
™
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4cell
5
state_spec*
Ы
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
	<layer*
j
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
I12
J13*
j
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
I12
J13*
* 
∞
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ptrace_0
Qtrace_1
Rtrace_2
Strace_3* 
6
Ttrace_0
Utrace_1
Vtrace_2
Wtrace_3* 
* 
№
Xiter

Ybeta_1

Zbeta_2
	[decay
\learning_rate=mш>mщ?mъ@mыAmьBmэCmюDm€EmАFmБGmВHmГImДJmЕ=vЖ>vЗ?vИ@vЙAvКBvЛCvМDvНEvОFvПGvРHvСIvТJvУ*

]serving_default* 

=0
>1
?2*

=0
>1
?2*
* 
Я

^states
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
dtrace_0
etrace_1
ftrace_2
gtrace_3* 
6
htrace_0
itrace_1
jtrace_2
ktrace_3* 
”
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses
r_random_generator

=kernel
>recurrent_kernel
?bias*
* 

@0
A1
B2*

@0
A1
B2*
* 
Я

sstates
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
ytrace_0
ztrace_1
{trace_2
|trace_3* 
7
}trace_0
~trace_1
trace_2
Аtrace_3* 
Џ
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses
З_random_generator

@kernel
Arecurrent_kernel
Bbias*
* 
* 
* 
* 
Ц
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 

Нtrace_0* 

Оtrace_0* 

C0
D1
E2*

C0
D1
E2*
* 
•
Пstates
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
:
Хtrace_0
Цtrace_1
Чtrace_2
Шtrace_3* 
:
Щtrace_0
Ъtrace_1
Ыtrace_2
Ьtrace_3* 
Џ
Э	variables
Юtrainable_variables
Яregularization_losses
†	keras_api
°__call__
+Ґ&call_and_return_all_conditional_losses
£_random_generator

Ckernel
Drecurrent_kernel
Ebias*
* 

F0
G1
H2*

F0
G1
H2*
* 
•
§states
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
:
™trace_0
Ђtrace_1
ђtrace_2
≠trace_3* 
:
Ѓtrace_0
ѓtrace_1
∞trace_2
±trace_3* 
Џ
≤	variables
≥trainable_variables
іregularization_losses
µ	keras_api
ґ__call__
+Ј&call_and_return_all_conditional_losses
Є_random_generator

Fkernel
Grecurrent_kernel
Hbias*
* 

I0
J1*

I0
J1*
* 
Ш
єnon_trainable_variables
Їlayers
їmetrics
 Љlayer_regularization_losses
љlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

Њtrace_0
њtrace_1* 

јtrace_0
Ѕtrace_1* 
ђ
¬	variables
√trainable_variables
ƒregularization_losses
≈	keras_api
∆__call__
+«&call_and_return_all_conditional_losses

Ikernel
Jbias*
e_
VARIABLE_VALUE%simple_rnn_8/simple_rnn_cell_8/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#simple_rnn_8/simple_rnn_cell_8/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_9/simple_rnn_cell_9/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#simple_rnn_9/simple_rnn_cell_9/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'simple_rnn_10/simple_rnn_cell_10/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1simple_rnn_10/simple_rnn_cell_10/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_10/simple_rnn_cell_10/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'simple_rnn_11/simple_rnn_cell_11/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1simple_rnn_11/simple_rnn_cell_11/recurrent_kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%simple_rnn_11/simple_rnn_cell_11/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEtime_distributed_18/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEtime_distributed_18/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

»0*
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

0*
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
=0
>1
?2*

=0
>1
?2*
* 
Ш
…non_trainable_variables
 layers
Ћmetrics
 ћlayer_regularization_losses
Ќlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses*

ќtrace_0
ѕtrace_1* 

–trace_0
—trace_1* 
* 
* 
* 

0*
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
@0
A1
B2*

@0
A1
B2*
* 
Ю
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses*

„trace_0
Ўtrace_1* 

ўtrace_0
Џtrace_1* 
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
C0
D1
E2*

C0
D1
E2*
* 
Ю
џnon_trainable_variables
№layers
Ёmetrics
 ёlayer_regularization_losses
яlayer_metrics
Э	variables
Юtrainable_variables
Яregularization_losses
°__call__
+Ґ&call_and_return_all_conditional_losses
'Ґ"call_and_return_conditional_losses*

аtrace_0
бtrace_1* 

вtrace_0
гtrace_1* 
* 
* 
* 

40*
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
F0
G1
H2*

F0
G1
H2*
* 
Ю
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
≤	variables
≥trainable_variables
іregularization_losses
ґ__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses*

йtrace_0
кtrace_1* 

лtrace_0
мtrace_1* 
* 
* 

<0*
* 
* 
* 
* 
* 
* 
* 

I0
J1*

I0
J1*
* 
Ю
нnon_trainable_variables
оlayers
пmetrics
 рlayer_regularization_losses
сlayer_metrics
¬	variables
√trainable_variables
ƒregularization_losses
∆__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses*

тtrace_0* 

уtrace_0* 
<
ф	variables
х	keras_api

цtotal

чcount*
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

ц0
ч1*

ф	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
УМ
VARIABLE_VALUE6Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUE*Adam/simple_rnn_8/simple_rnn_cell_8/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
УМ
VARIABLE_VALUE6Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUE*Adam/simple_rnn_9/simple_rnn_cell_9/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUE.Adam/simple_rnn_10/simple_rnn_cell_10/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ХО
VARIABLE_VALUE8Adam/simple_rnn_10/simple_rnn_cell_10/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/simple_rnn_10/simple_rnn_cell_10/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUE.Adam/simple_rnn_11/simple_rnn_cell_11/kernel/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЦП
VARIABLE_VALUE8Adam/simple_rnn_11/simple_rnn_cell_11/recurrent_kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUE,Adam/simple_rnn_11/simple_rnn_cell_11/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/time_distributed_18/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/time_distributed_18/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
УМ
VARIABLE_VALUE6Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUE*Adam/simple_rnn_8/simple_rnn_cell_8/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
УМ
VARIABLE_VALUE6Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUE*Adam/simple_rnn_9/simple_rnn_cell_9/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUE.Adam/simple_rnn_10/simple_rnn_cell_10/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ХО
VARIABLE_VALUE8Adam/simple_rnn_10/simple_rnn_cell_10/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/simple_rnn_10/simple_rnn_cell_10/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUE.Adam/simple_rnn_11/simple_rnn_cell_11/kernel/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЦП
VARIABLE_VALUE8Adam/simple_rnn_11/simple_rnn_cell_11/recurrent_kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUE,Adam/simple_rnn_11/simple_rnn_cell_11/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/time_distributed_18/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/time_distributed_18/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Џ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename9simple_rnn_8/simple_rnn_cell_8/kernel/Read/ReadVariableOpCsimple_rnn_8/simple_rnn_cell_8/recurrent_kernel/Read/ReadVariableOp7simple_rnn_8/simple_rnn_cell_8/bias/Read/ReadVariableOp9simple_rnn_9/simple_rnn_cell_9/kernel/Read/ReadVariableOpCsimple_rnn_9/simple_rnn_cell_9/recurrent_kernel/Read/ReadVariableOp7simple_rnn_9/simple_rnn_cell_9/bias/Read/ReadVariableOp;simple_rnn_10/simple_rnn_cell_10/kernel/Read/ReadVariableOpEsimple_rnn_10/simple_rnn_cell_10/recurrent_kernel/Read/ReadVariableOp9simple_rnn_10/simple_rnn_cell_10/bias/Read/ReadVariableOp;simple_rnn_11/simple_rnn_cell_11/kernel/Read/ReadVariableOpEsimple_rnn_11/simple_rnn_cell_11/recurrent_kernel/Read/ReadVariableOp9simple_rnn_11/simple_rnn_cell_11/bias/Read/ReadVariableOp.time_distributed_18/kernel/Read/ReadVariableOp,time_distributed_18/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp@Adam/simple_rnn_8/simple_rnn_cell_8/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_8/simple_rnn_cell_8/bias/m/Read/ReadVariableOp@Adam/simple_rnn_9/simple_rnn_cell_9/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_9/simple_rnn_cell_9/bias/m/Read/ReadVariableOpBAdam/simple_rnn_10/simple_rnn_cell_10/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_10/simple_rnn_cell_10/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_10/simple_rnn_cell_10/bias/m/Read/ReadVariableOpBAdam/simple_rnn_11/simple_rnn_cell_11/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_11/simple_rnn_cell_11/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_11/simple_rnn_cell_11/bias/m/Read/ReadVariableOp5Adam/time_distributed_18/kernel/m/Read/ReadVariableOp3Adam/time_distributed_18/bias/m/Read/ReadVariableOp@Adam/simple_rnn_8/simple_rnn_cell_8/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_8/simple_rnn_cell_8/bias/v/Read/ReadVariableOp@Adam/simple_rnn_9/simple_rnn_cell_9/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_9/simple_rnn_cell_9/bias/v/Read/ReadVariableOpBAdam/simple_rnn_10/simple_rnn_cell_10/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_10/simple_rnn_cell_10/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_10/simple_rnn_cell_10/bias/v/Read/ReadVariableOpBAdam/simple_rnn_11/simple_rnn_cell_11/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_11/simple_rnn_cell_11/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_11/simple_rnn_cell_11/bias/v/Read/ReadVariableOp5Adam/time_distributed_18/kernel/v/Read/ReadVariableOp3Adam/time_distributed_18/bias/v/Read/ReadVariableOpConst*>
Tin7
523	*
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
 __inference__traced_save_1740655
Б
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename%simple_rnn_8/simple_rnn_cell_8/kernel/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel#simple_rnn_8/simple_rnn_cell_8/bias%simple_rnn_9/simple_rnn_cell_9/kernel/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel#simple_rnn_9/simple_rnn_cell_9/bias'simple_rnn_10/simple_rnn_cell_10/kernel1simple_rnn_10/simple_rnn_cell_10/recurrent_kernel%simple_rnn_10/simple_rnn_cell_10/bias'simple_rnn_11/simple_rnn_cell_11/kernel1simple_rnn_11/simple_rnn_cell_11/recurrent_kernel%simple_rnn_11/simple_rnn_cell_11/biastime_distributed_18/kerneltime_distributed_18/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/m6Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/m*Adam/simple_rnn_8/simple_rnn_cell_8/bias/m,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/m6Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/m*Adam/simple_rnn_9/simple_rnn_cell_9/bias/m.Adam/simple_rnn_10/simple_rnn_cell_10/kernel/m8Adam/simple_rnn_10/simple_rnn_cell_10/recurrent_kernel/m,Adam/simple_rnn_10/simple_rnn_cell_10/bias/m.Adam/simple_rnn_11/simple_rnn_cell_11/kernel/m8Adam/simple_rnn_11/simple_rnn_cell_11/recurrent_kernel/m,Adam/simple_rnn_11/simple_rnn_cell_11/bias/m!Adam/time_distributed_18/kernel/mAdam/time_distributed_18/bias/m,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/v6Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/v*Adam/simple_rnn_8/simple_rnn_cell_8/bias/v,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/v6Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/v*Adam/simple_rnn_9/simple_rnn_cell_9/bias/v.Adam/simple_rnn_10/simple_rnn_cell_10/kernel/v8Adam/simple_rnn_10/simple_rnn_cell_10/recurrent_kernel/v,Adam/simple_rnn_10/simple_rnn_cell_10/bias/v.Adam/simple_rnn_11/simple_rnn_cell_11/kernel/v8Adam/simple_rnn_11/simple_rnn_cell_11/recurrent_kernel/v,Adam/simple_rnn_11/simple_rnn_cell_11/bias/v!Adam/time_distributed_18/kernel/vAdam/time_distributed_18/bias/v*=
Tin6
422*
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
#__inference__traced_restore_1740812Бф2
 =
√
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1740050

inputsC
1simple_rnn_cell_11_matmul_readvariableop_resource: @
2simple_rnn_cell_11_biasadd_readvariableop_resource: E
3simple_rnn_cell_11_matmul_1_readvariableop_resource:  
identityИҐ)simple_rnn_cell_11/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_11/MatMul/ReadVariableOpҐ*simple_rnn_cell_11/MatMul_1/ReadVariableOpҐwhile;
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
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
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
valueB"€€€€   а
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
:€€€€€€€€€*
shrink_axis_maskЪ
(simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

: *
dtype0°
simple_rnn_cell_11/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ш
)simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ѓ
simple_rnn_cell_11/BiasAddBiasAdd#simple_rnn_cell_11/MatMul:product:01simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ю
*simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0Ы
simple_rnn_cell_11/MatMul_1MatMulzeros:output:02simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Э
simple_rnn_cell_11/addAddV2#simple_rnn_cell_11/BiasAdd:output:0%simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ m
simple_rnn_cell_11/ReluRelusimple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_11_matmul_readvariableop_resource2simple_rnn_cell_11_biasadd_readvariableop_resource3simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1739984*
condR
while_cond_1739983*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype0h
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
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ “
NoOpNoOp*^simple_rnn_cell_11/BiasAdd/ReadVariableOp)^simple_rnn_cell_11/MatMul/ReadVariableOp+^simple_rnn_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2V
)simple_rnn_cell_11/BiasAdd/ReadVariableOp)simple_rnn_cell_11/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_11/MatMul/ReadVariableOp(simple_rnn_cell_11/MatMul/ReadVariableOp2X
*simple_rnn_cell_11/MatMul_1/ReadVariableOp*simple_rnn_cell_11/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
љ

№
4__inference_simple_rnn_cell_11_layer_call_fn_1740432

inputs
states_0
unknown: 
	unknown_0: 
	unknown_1:  
identity

identity_1ИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_1735795o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/0
И
Є
.__inference_simple_rnn_8_layer_call_fn_1738266

inputs
unknown: 
	unknown_0: 
	unknown_1:  
identityИҐStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1736114s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ш9
ц
 simple_rnn_11_while_body_17381558
4simple_rnn_11_while_simple_rnn_11_while_loop_counter>
:simple_rnn_11_while_simple_rnn_11_while_maximum_iterations#
simple_rnn_11_while_placeholder%
!simple_rnn_11_while_placeholder_1%
!simple_rnn_11_while_placeholder_27
3simple_rnn_11_while_simple_rnn_11_strided_slice_1_0s
osimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_11_while_simple_rnn_cell_11_matmul_readvariableop_resource_0: V
Hsimple_rnn_11_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0: [
Isimple_rnn_11_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:   
simple_rnn_11_while_identity"
simple_rnn_11_while_identity_1"
simple_rnn_11_while_identity_2"
simple_rnn_11_while_identity_3"
simple_rnn_11_while_identity_45
1simple_rnn_11_while_simple_rnn_11_strided_slice_1q
msimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_11_while_simple_rnn_cell_11_matmul_readvariableop_resource: T
Fsimple_rnn_11_while_simple_rnn_cell_11_biasadd_readvariableop_resource: Y
Gsimple_rnn_11_while_simple_rnn_cell_11_matmul_1_readvariableop_resource:  ИҐ=simple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpҐ<simple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOpҐ>simple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpЦ
Esimple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   м
7simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_11_while_placeholderNsimple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0ƒ
<simple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_11_while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0п
-simple_rnn_11/while/simple_rnn_cell_11/MatMulMatMul>simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ¬
=simple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_11_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0л
.simple_rnn_11/while/simple_rnn_cell_11/BiasAddBiasAdd7simple_rnn_11/while/simple_rnn_cell_11/MatMul:product:0Esimple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ »
>simple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_11_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0÷
/simple_rnn_11/while/simple_rnn_cell_11/MatMul_1MatMul!simple_rnn_11_while_placeholder_2Fsimple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ў
*simple_rnn_11/while/simple_rnn_cell_11/addAddV27simple_rnn_11/while/simple_rnn_cell_11/BiasAdd:output:09simple_rnn_11/while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
+simple_rnn_11/while/simple_rnn_cell_11/ReluRelu.simple_rnn_11/while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ М
8simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_11_while_placeholder_1simple_rnn_11_while_placeholder9simple_rnn_11/while/simple_rnn_cell_11/Relu:activations:0*
_output_shapes
: *
element_dtype0:йи“[
simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ж
simple_rnn_11/while/addAddV2simple_rnn_11_while_placeholder"simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Я
simple_rnn_11/while/add_1AddV24simple_rnn_11_while_simple_rnn_11_while_loop_counter$simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: Г
simple_rnn_11/while/IdentityIdentitysimple_rnn_11/while/add_1:z:0^simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Ґ
simple_rnn_11/while/Identity_1Identity:simple_rnn_11_while_simple_rnn_11_while_maximum_iterations^simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Г
simple_rnn_11/while/Identity_2Identitysimple_rnn_11/while/add:z:0^simple_rnn_11/while/NoOp*
T0*
_output_shapes
: ∞
simple_rnn_11/while/Identity_3IdentityHsimple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_11/while/NoOp*
T0*
_output_shapes
: ≤
simple_rnn_11/while/Identity_4Identity9simple_rnn_11/while/simple_rnn_cell_11/Relu:activations:0^simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Ъ
simple_rnn_11/while/NoOpNoOp>^simple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp=^simple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOp?^simple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_11_while_identity%simple_rnn_11/while/Identity:output:0"I
simple_rnn_11_while_identity_1'simple_rnn_11/while/Identity_1:output:0"I
simple_rnn_11_while_identity_2'simple_rnn_11/while/Identity_2:output:0"I
simple_rnn_11_while_identity_3'simple_rnn_11/while/Identity_3:output:0"I
simple_rnn_11_while_identity_4'simple_rnn_11/while/Identity_4:output:0"h
1simple_rnn_11_while_simple_rnn_11_strided_slice_13simple_rnn_11_while_simple_rnn_11_strided_slice_1_0"Т
Fsimple_rnn_11_while_simple_rnn_cell_11_biasadd_readvariableop_resourceHsimple_rnn_11_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"Ф
Gsimple_rnn_11_while_simple_rnn_cell_11_matmul_1_readvariableop_resourceIsimple_rnn_11_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"Р
Esimple_rnn_11_while_simple_rnn_cell_11_matmul_readvariableop_resourceGsimple_rnn_11_while_simple_rnn_cell_11_matmul_readvariableop_resource_0"а
msimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorosimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2~
=simple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp=simple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2|
<simple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOp<simple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOp2А
>simple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp>simple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
я
ѓ
while_cond_1739015
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1739015___redundant_placeholder05
1while_while_cond_1739015___redundant_placeholder15
1while_while_cond_1739015___redundant_placeholder25
1while_while_cond_1739015___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Ш
Ї
.__inference_simple_rnn_9_layer_call_fn_1738720
inputs_0
unknown: 
	unknown_0:
	unknown_1:
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1735150o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
inputs/0
м,
“
while_body_1739984
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_11_matmul_readvariableop_resource_0: H
:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0: M
;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_11_matmul_readvariableop_resource: F
8while_simple_rnn_cell_11_biasadd_readvariableop_resource: K
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource:  ИҐ/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_11/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0®
.while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0≈
while/simple_rnn_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ¶
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ѕ
 while/simple_rnn_cell_11/BiasAddBiasAdd)while/simple_rnn_cell_11/MatMul:product:07while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ђ
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0ђ
!while/simple_rnn_cell_11/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ѓ
while/simple_rnn_cell_11/addAddV2)while/simple_rnn_cell_11/BiasAdd:output:0+while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ y
while/simple_rnn_cell_11/ReluRelu while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ‘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder+while/simple_rnn_cell_11/Relu:activations:0*
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
: И
while/Identity_4Identity+while/simple_rnn_cell_11/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ в

while/NoOpNoOp0^while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_11/MatMul/ReadVariableOp1^while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_11_matmul_readvariableop_resource9while_simple_rnn_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2b
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_11/MatMul/ReadVariableOp.while/simple_rnn_cell_11/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
ќ>
Њ
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1738863
inputs_0B
0simple_rnn_cell_9_matmul_readvariableop_resource: ?
1simple_rnn_cell_9_biasadd_readvariableop_resource:D
2simple_rnn_cell_9_matmul_1_readvariableop_resource:
identityИҐ(simple_rnn_cell_9/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_9/MatMul/ReadVariableOpҐ)simple_rnn_cell_9/MatMul_1/ReadVariableOpҐwhile=
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
value	B :s
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ D
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
valueB"€€€€    а
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
:€€€€€€€€€ *
shrink_axis_maskШ
'simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Я
simple_rnn_cell_9/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ц
(simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ђ
simple_rnn_cell_9/BiasAddBiasAdd"simple_rnn_cell_9/MatMul:product:00simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
)simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Щ
simple_rnn_cell_9/MatMul_1MatMulzeros:output:01simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
simple_rnn_cell_9/addAddV2"simple_rnn_cell_9/BiasAdd:output:0$simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€k
simple_rnn_cell_9/ReluRelusimple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_9_matmul_readvariableop_resource1simple_rnn_cell_9_biasadd_readvariableop_resource2simple_rnn_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1738796*
condR
while_cond_1738795*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
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
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ѕ
NoOpNoOp)^simple_rnn_cell_9/BiasAdd/ReadVariableOp(^simple_rnn_cell_9/MatMul/ReadVariableOp*^simple_rnn_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 2T
(simple_rnn_cell_9/BiasAdd/ReadVariableOp(simple_rnn_cell_9/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_9/MatMul/ReadVariableOp'simple_rnn_cell_9/MatMul/ReadVariableOp2V
)simple_rnn_cell_9/MatMul_1/ReadVariableOp)simple_rnn_cell_9/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
inputs/0
д
л
/__inference_sequential_19_layer_call_fn_1737324

inputs
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8: 
	unknown_9: 

unknown_10:  

unknown_11: 

unknown_12:
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_19_layer_call_and_return_conditional_losses_1736478s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
 =
√
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1740158

inputsC
1simple_rnn_cell_11_matmul_readvariableop_resource: @
2simple_rnn_cell_11_biasadd_readvariableop_resource: E
3simple_rnn_cell_11_matmul_1_readvariableop_resource:  
identityИҐ)simple_rnn_cell_11/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_11/MatMul/ReadVariableOpҐ*simple_rnn_cell_11/MatMul_1/ReadVariableOpҐwhile;
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
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
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
valueB"€€€€   а
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
:€€€€€€€€€*
shrink_axis_maskЪ
(simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

: *
dtype0°
simple_rnn_cell_11/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ш
)simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ѓ
simple_rnn_cell_11/BiasAddBiasAdd#simple_rnn_cell_11/MatMul:product:01simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ю
*simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0Ы
simple_rnn_cell_11/MatMul_1MatMulzeros:output:02simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Э
simple_rnn_cell_11/addAddV2#simple_rnn_cell_11/BiasAdd:output:0%simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ m
simple_rnn_cell_11/ReluRelusimple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_11_matmul_readvariableop_resource2simple_rnn_cell_11_biasadd_readvariableop_resource3simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1740092*
condR
while_cond_1740091*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype0h
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
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ “
NoOpNoOp*^simple_rnn_cell_11/BiasAdd/ReadVariableOp)^simple_rnn_cell_11/MatMul/ReadVariableOp+^simple_rnn_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2V
)simple_rnn_cell_11/BiasAdd/ReadVariableOp)simple_rnn_cell_11/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_11/MatMul/ReadVariableOp(simple_rnn_cell_11/MatMul/ReadVariableOp2X
*simple_rnn_cell_11/MatMul_1/ReadVariableOp*simple_rnn_cell_11/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
И
Є
.__inference_simple_rnn_8_layer_call_fn_1738277

inputs
unknown: 
	unknown_0: 
	unknown_1:  
identityИҐStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1737022s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
И>
≈
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1739358
inputs_0C
1simple_rnn_cell_10_matmul_readvariableop_resource:@
2simple_rnn_cell_10_biasadd_readvariableop_resource:E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:
identityИҐ)simple_rnn_cell_10/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_10/MatMul/ReadVariableOpҐ*simple_rnn_cell_10/MatMul_1/ReadVariableOpҐwhile=
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
value	B :s
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
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
valueB"€€€€   а
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
:€€€€€€€€€*
shrink_axis_maskЪ
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype0°
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Ы
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€m
simple_rnn_cell_10/ReluRelusimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1739292*
condR
while_cond_1739291*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
element_dtype0h
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
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€“
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
Џ

є
 simple_rnn_10_while_cond_17380508
4simple_rnn_10_while_simple_rnn_10_while_loop_counter>
:simple_rnn_10_while_simple_rnn_10_while_maximum_iterations#
simple_rnn_10_while_placeholder%
!simple_rnn_10_while_placeholder_1%
!simple_rnn_10_while_placeholder_2:
6simple_rnn_10_while_less_simple_rnn_10_strided_slice_1Q
Msimple_rnn_10_while_simple_rnn_10_while_cond_1738050___redundant_placeholder0Q
Msimple_rnn_10_while_simple_rnn_10_while_cond_1738050___redundant_placeholder1Q
Msimple_rnn_10_while_simple_rnn_10_while_cond_1738050___redundant_placeholder2Q
Msimple_rnn_10_while_simple_rnn_10_while_cond_1738050___redundant_placeholder3 
simple_rnn_10_while_identity
Ъ
simple_rnn_10/while/LessLesssimple_rnn_10_while_placeholder6simple_rnn_10_while_less_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_10/while/IdentityIdentitysimple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_10_while_identity%simple_rnn_10/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
™4
§
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1735459

inputs,
simple_rnn_cell_10_1735384:(
simple_rnn_cell_10_1735386:,
simple_rnn_cell_10_1735388:
identityИҐ*simple_rnn_cell_10/StatefulPartitionedCallҐwhile;
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
value	B :s
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
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
valueB"€€€€   а
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
:€€€€€€€€€*
shrink_axis_maskр
*simple_rnn_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_10_1735384simple_rnn_cell_10_1735386simple_rnn_cell_10_1735388*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_1735383n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Х
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_10_1735384simple_rnn_cell_10_1735386simple_rnn_cell_10_1735388*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1735396*
condR
while_cond_1735395*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
element_dtype0h
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
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€{
NoOpNoOp+^simple_rnn_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2X
*simple_rnn_cell_10/StatefulPartitionedCall*simple_rnn_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
А
Є
.__inference_simple_rnn_9_layer_call_fn_1738742

inputs
unknown: 
	unknown_0:
	unknown_1:
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1736231o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
я
ѓ
while_cond_1736163
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1736163___redundant_placeholder05
1while_while_cond_1736163___redundant_placeholder15
1while_while_cond_1736163___redundant_placeholder25
1while_while_cond_1736163___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Ђ>
Љ
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1736892

inputsB
0simple_rnn_cell_9_matmul_readvariableop_resource: ?
1simple_rnn_cell_9_biasadd_readvariableop_resource:D
2simple_rnn_cell_9_matmul_1_readvariableop_resource:
identityИҐ(simple_rnn_cell_9/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_9/MatMul/ReadVariableOpҐ)simple_rnn_cell_9/MatMul_1/ReadVariableOpҐwhile;
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
value	B :s
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ D
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
valueB"€€€€    а
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
:€€€€€€€€€ *
shrink_axis_maskШ
'simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Я
simple_rnn_cell_9/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ц
(simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ђ
simple_rnn_cell_9/BiasAddBiasAdd"simple_rnn_cell_9/MatMul:product:00simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
)simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Щ
simple_rnn_cell_9/MatMul_1MatMulzeros:output:01simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
simple_rnn_cell_9/addAddV2"simple_rnn_cell_9/BiasAdd:output:0$simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€k
simple_rnn_cell_9/ReluRelusimple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_9_matmul_readvariableop_resource1simple_rnn_cell_9_biasadd_readvariableop_resource2simple_rnn_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1736825*
condR
while_cond_1736824*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
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
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ѕ
NoOpNoOp)^simple_rnn_cell_9/BiasAdd/ReadVariableOp(^simple_rnn_cell_9/MatMul/ReadVariableOp*^simple_rnn_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ : : : 2T
(simple_rnn_cell_9/BiasAdd/ReadVariableOp(simple_rnn_cell_9/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_9/MatMul/ReadVariableOp'simple_rnn_cell_9/MatMul/ReadVariableOp2V
)simple_rnn_cell_9/MatMul_1/ReadVariableOp)simple_rnn_cell_9/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Пр
й
"__inference__wrapped_model_1734732
simple_rnn_8_input]
Ksequential_19_simple_rnn_8_simple_rnn_cell_8_matmul_readvariableop_resource: Z
Lsequential_19_simple_rnn_8_simple_rnn_cell_8_biasadd_readvariableop_resource: _
Msequential_19_simple_rnn_8_simple_rnn_cell_8_matmul_1_readvariableop_resource:  ]
Ksequential_19_simple_rnn_9_simple_rnn_cell_9_matmul_readvariableop_resource: Z
Lsequential_19_simple_rnn_9_simple_rnn_cell_9_biasadd_readvariableop_resource:_
Msequential_19_simple_rnn_9_simple_rnn_cell_9_matmul_1_readvariableop_resource:_
Msequential_19_simple_rnn_10_simple_rnn_cell_10_matmul_readvariableop_resource:\
Nsequential_19_simple_rnn_10_simple_rnn_cell_10_biasadd_readvariableop_resource:a
Osequential_19_simple_rnn_10_simple_rnn_cell_10_matmul_1_readvariableop_resource:_
Msequential_19_simple_rnn_11_simple_rnn_cell_11_matmul_readvariableop_resource: \
Nsequential_19_simple_rnn_11_simple_rnn_cell_11_biasadd_readvariableop_resource: a
Osequential_19_simple_rnn_11_simple_rnn_cell_11_matmul_1_readvariableop_resource:  [
Isequential_19_time_distributed_18_dense_21_matmul_readvariableop_resource: X
Jsequential_19_time_distributed_18_dense_21_biasadd_readvariableop_resource:
identityИҐEsequential_19/simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOpҐDsequential_19/simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOpҐFsequential_19/simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOpҐ!sequential_19/simple_rnn_10/whileҐEsequential_19/simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOpҐDsequential_19/simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOpҐFsequential_19/simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOpҐ!sequential_19/simple_rnn_11/whileҐCsequential_19/simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOpҐBsequential_19/simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOpҐDsequential_19/simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOpҐ sequential_19/simple_rnn_8/whileҐCsequential_19/simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOpҐBsequential_19/simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOpҐDsequential_19/simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOpҐ sequential_19/simple_rnn_9/whileҐAsequential_19/time_distributed_18/dense_21/BiasAdd/ReadVariableOpҐ@sequential_19/time_distributed_18/dense_21/MatMul/ReadVariableOpb
 sequential_19/simple_rnn_8/ShapeShapesimple_rnn_8_input*
T0*
_output_shapes
:x
.sequential_19/simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential_19/simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential_19/simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
(sequential_19/simple_rnn_8/strided_sliceStridedSlice)sequential_19/simple_rnn_8/Shape:output:07sequential_19/simple_rnn_8/strided_slice/stack:output:09sequential_19/simple_rnn_8/strided_slice/stack_1:output:09sequential_19/simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_19/simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ƒ
'sequential_19/simple_rnn_8/zeros/packedPack1sequential_19/simple_rnn_8/strided_slice:output:02sequential_19/simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&sequential_19/simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    љ
 sequential_19/simple_rnn_8/zerosFill0sequential_19/simple_rnn_8/zeros/packed:output:0/sequential_19/simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ~
)sequential_19/simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ѓ
$sequential_19/simple_rnn_8/transpose	Transposesimple_rnn_8_input2sequential_19/simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€z
"sequential_19/simple_rnn_8/Shape_1Shape(sequential_19/simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:z
0sequential_19/simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_19/simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_19/simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
*sequential_19/simple_rnn_8/strided_slice_1StridedSlice+sequential_19/simple_rnn_8/Shape_1:output:09sequential_19/simple_rnn_8/strided_slice_1/stack:output:0;sequential_19/simple_rnn_8/strided_slice_1/stack_1:output:0;sequential_19/simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
6sequential_19/simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Е
(sequential_19/simple_rnn_8/TensorArrayV2TensorListReserve?sequential_19/simple_rnn_8/TensorArrayV2/element_shape:output:03sequential_19/simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“°
Psequential_19/simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ±
Bsequential_19/simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_19/simple_rnn_8/transpose:y:0Ysequential_19/simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“z
0sequential_19/simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_19/simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_19/simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
*sequential_19/simple_rnn_8/strided_slice_2StridedSlice(sequential_19/simple_rnn_8/transpose:y:09sequential_19/simple_rnn_8/strided_slice_2/stack:output:0;sequential_19/simple_rnn_8/strided_slice_2/stack_1:output:0;sequential_19/simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskќ
Bsequential_19/simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOpKsequential_19_simple_rnn_8_simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

: *
dtype0р
3sequential_19/simple_rnn_8/simple_rnn_cell_8/MatMulMatMul3sequential_19/simple_rnn_8/strided_slice_2:output:0Jsequential_19/simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ћ
Csequential_19/simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOpLsequential_19_simple_rnn_8_simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0э
4sequential_19/simple_rnn_8/simple_rnn_cell_8/BiasAddBiasAdd=sequential_19/simple_rnn_8/simple_rnn_cell_8/MatMul:product:0Ksequential_19/simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ “
Dsequential_19/simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOpMsequential_19_simple_rnn_8_simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0к
5sequential_19/simple_rnn_8/simple_rnn_cell_8/MatMul_1MatMul)sequential_19/simple_rnn_8/zeros:output:0Lsequential_19/simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ л
0sequential_19/simple_rnn_8/simple_rnn_cell_8/addAddV2=sequential_19/simple_rnn_8/simple_rnn_cell_8/BiasAdd:output:0?sequential_19/simple_rnn_8/simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ °
1sequential_19/simple_rnn_8/simple_rnn_cell_8/ReluRelu4sequential_19/simple_rnn_8/simple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Й
8sequential_19/simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Й
*sequential_19/simple_rnn_8/TensorArrayV2_1TensorListReserveAsequential_19/simple_rnn_8/TensorArrayV2_1/element_shape:output:03sequential_19/simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“a
sequential_19/simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3sequential_19/simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€o
-sequential_19/simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : є
 sequential_19/simple_rnn_8/whileWhile6sequential_19/simple_rnn_8/while/loop_counter:output:0<sequential_19/simple_rnn_8/while/maximum_iterations:output:0(sequential_19/simple_rnn_8/time:output:03sequential_19/simple_rnn_8/TensorArrayV2_1:handle:0)sequential_19/simple_rnn_8/zeros:output:03sequential_19/simple_rnn_8/strided_slice_1:output:0Rsequential_19/simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ksequential_19_simple_rnn_8_simple_rnn_cell_8_matmul_readvariableop_resourceLsequential_19_simple_rnn_8_simple_rnn_cell_8_biasadd_readvariableop_resourceMsequential_19_simple_rnn_8_simple_rnn_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *9
body1R/
-sequential_19_simple_rnn_8_while_body_1734336*9
cond1R/
-sequential_19_simple_rnn_8_while_cond_1734335*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Ь
Ksequential_19/simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    У
=sequential_19/simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_19/simple_rnn_8/while:output:3Tsequential_19/simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype0Г
0sequential_19/simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€|
2sequential_19/simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2sequential_19/simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
*sequential_19/simple_rnn_8/strided_slice_3StridedSliceFsequential_19/simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:09sequential_19/simple_rnn_8/strided_slice_3/stack:output:0;sequential_19/simple_rnn_8/strided_slice_3/stack_1:output:0;sequential_19/simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maskА
+sequential_19/simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          з
&sequential_19/simple_rnn_8/transpose_1	TransposeFsequential_19/simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:04sequential_19/simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ z
 sequential_19/simple_rnn_9/ShapeShape*sequential_19/simple_rnn_8/transpose_1:y:0*
T0*
_output_shapes
:x
.sequential_19/simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential_19/simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential_19/simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
(sequential_19/simple_rnn_9/strided_sliceStridedSlice)sequential_19/simple_rnn_9/Shape:output:07sequential_19/simple_rnn_9/strided_slice/stack:output:09sequential_19/simple_rnn_9/strided_slice/stack_1:output:09sequential_19/simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_19/simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ƒ
'sequential_19/simple_rnn_9/zeros/packedPack1sequential_19/simple_rnn_9/strided_slice:output:02sequential_19/simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&sequential_19/simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    љ
 sequential_19/simple_rnn_9/zerosFill0sequential_19/simple_rnn_9/zeros/packed:output:0/sequential_19/simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€~
)sequential_19/simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
$sequential_19/simple_rnn_9/transpose	Transpose*sequential_19/simple_rnn_8/transpose_1:y:02sequential_19/simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ z
"sequential_19/simple_rnn_9/Shape_1Shape(sequential_19/simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:z
0sequential_19/simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_19/simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_19/simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
*sequential_19/simple_rnn_9/strided_slice_1StridedSlice+sequential_19/simple_rnn_9/Shape_1:output:09sequential_19/simple_rnn_9/strided_slice_1/stack:output:0;sequential_19/simple_rnn_9/strided_slice_1/stack_1:output:0;sequential_19/simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
6sequential_19/simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Е
(sequential_19/simple_rnn_9/TensorArrayV2TensorListReserve?sequential_19/simple_rnn_9/TensorArrayV2/element_shape:output:03sequential_19/simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“°
Psequential_19/simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ±
Bsequential_19/simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_19/simple_rnn_9/transpose:y:0Ysequential_19/simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“z
0sequential_19/simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_19/simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_19/simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
*sequential_19/simple_rnn_9/strided_slice_2StridedSlice(sequential_19/simple_rnn_9/transpose:y:09sequential_19/simple_rnn_9/strided_slice_2/stack:output:0;sequential_19/simple_rnn_9/strided_slice_2/stack_1:output:0;sequential_19/simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maskќ
Bsequential_19/simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOpKsequential_19_simple_rnn_9_simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype0р
3sequential_19/simple_rnn_9/simple_rnn_cell_9/MatMulMatMul3sequential_19/simple_rnn_9/strided_slice_2:output:0Jsequential_19/simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ћ
Csequential_19/simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOpLsequential_19_simple_rnn_9_simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0э
4sequential_19/simple_rnn_9/simple_rnn_cell_9/BiasAddBiasAdd=sequential_19/simple_rnn_9/simple_rnn_cell_9/MatMul:product:0Ksequential_19/simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€“
Dsequential_19/simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOpMsequential_19_simple_rnn_9_simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0к
5sequential_19/simple_rnn_9/simple_rnn_cell_9/MatMul_1MatMul)sequential_19/simple_rnn_9/zeros:output:0Lsequential_19/simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€л
0sequential_19/simple_rnn_9/simple_rnn_cell_9/addAddV2=sequential_19/simple_rnn_9/simple_rnn_cell_9/BiasAdd:output:0?sequential_19/simple_rnn_9/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€°
1sequential_19/simple_rnn_9/simple_rnn_cell_9/ReluRelu4sequential_19/simple_rnn_9/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Й
8sequential_19/simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   y
7sequential_19/simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ц
*sequential_19/simple_rnn_9/TensorArrayV2_1TensorListReserveAsequential_19/simple_rnn_9/TensorArrayV2_1/element_shape:output:0@sequential_19/simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“a
sequential_19/simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3sequential_19/simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€o
-sequential_19/simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : є
 sequential_19/simple_rnn_9/whileWhile6sequential_19/simple_rnn_9/while/loop_counter:output:0<sequential_19/simple_rnn_9/while/maximum_iterations:output:0(sequential_19/simple_rnn_9/time:output:03sequential_19/simple_rnn_9/TensorArrayV2_1:handle:0)sequential_19/simple_rnn_9/zeros:output:03sequential_19/simple_rnn_9/strided_slice_1:output:0Rsequential_19/simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ksequential_19_simple_rnn_9_simple_rnn_cell_9_matmul_readvariableop_resourceLsequential_19_simple_rnn_9_simple_rnn_cell_9_biasadd_readvariableop_resourceMsequential_19_simple_rnn_9_simple_rnn_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *9
body1R/
-sequential_19_simple_rnn_9_while_body_1734441*9
cond1R/
-sequential_19_simple_rnn_9_while_cond_1734440*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Ь
Ksequential_19/simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   І
=sequential_19/simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_19/simple_rnn_9/while:output:3Tsequential_19/simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0*
num_elementsГ
0sequential_19/simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€|
2sequential_19/simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2sequential_19/simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
*sequential_19/simple_rnn_9/strided_slice_3StridedSliceFsequential_19/simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:09sequential_19/simple_rnn_9/strided_slice_3/stack:output:0;sequential_19/simple_rnn_9/strided_slice_3/stack_1:output:0;sequential_19/simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskА
+sequential_19/simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          з
&sequential_19/simple_rnn_9/transpose_1	TransposeFsequential_19/simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:04sequential_19/simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€o
-sequential_19/repeat_vector_18/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Џ
)sequential_19/repeat_vector_18/ExpandDims
ExpandDims3sequential_19/simple_rnn_9/strided_slice_3:output:06sequential_19/repeat_vector_18/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€y
$sequential_19/repeat_vector_18/stackConst*
_output_shapes
:*
dtype0*!
valueB"         ƒ
#sequential_19/repeat_vector_18/TileTile2sequential_19/repeat_vector_18/ExpandDims:output:0-sequential_19/repeat_vector_18/stack:output:0*
T0*+
_output_shapes
:€€€€€€€€€}
!sequential_19/simple_rnn_10/ShapeShape,sequential_19/repeat_vector_18/Tile:output:0*
T0*
_output_shapes
:y
/sequential_19/simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_19/simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_19/simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
)sequential_19/simple_rnn_10/strided_sliceStridedSlice*sequential_19/simple_rnn_10/Shape:output:08sequential_19/simple_rnn_10/strided_slice/stack:output:0:sequential_19/simple_rnn_10/strided_slice/stack_1:output:0:sequential_19/simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*sequential_19/simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :«
(sequential_19/simple_rnn_10/zeros/packedPack2sequential_19/simple_rnn_10/strided_slice:output:03sequential_19/simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:l
'sequential_19/simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ј
!sequential_19/simple_rnn_10/zerosFill1sequential_19/simple_rnn_10/zeros/packed:output:00sequential_19/simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€
*sequential_19/simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ћ
%sequential_19/simple_rnn_10/transpose	Transpose,sequential_19/repeat_vector_18/Tile:output:03sequential_19/simple_rnn_10/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€|
#sequential_19/simple_rnn_10/Shape_1Shape)sequential_19/simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:{
1sequential_19/simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_19/simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_19/simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+sequential_19/simple_rnn_10/strided_slice_1StridedSlice,sequential_19/simple_rnn_10/Shape_1:output:0:sequential_19/simple_rnn_10/strided_slice_1/stack:output:0<sequential_19/simple_rnn_10/strided_slice_1/stack_1:output:0<sequential_19/simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskВ
7sequential_19/simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€И
)sequential_19/simple_rnn_10/TensorArrayV2TensorListReserve@sequential_19/simple_rnn_10/TensorArrayV2/element_shape:output:04sequential_19/simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ґ
Qsequential_19/simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   і
Csequential_19/simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)sequential_19/simple_rnn_10/transpose:y:0Zsequential_19/simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“{
1sequential_19/simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_19/simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_19/simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+sequential_19/simple_rnn_10/strided_slice_2StridedSlice)sequential_19/simple_rnn_10/transpose:y:0:sequential_19/simple_rnn_10/strided_slice_2/stack:output:0<sequential_19/simple_rnn_10/strided_slice_2/stack_1:output:0<sequential_19/simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask“
Dsequential_19/simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpMsequential_19_simple_rnn_10_simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype0х
5sequential_19/simple_rnn_10/simple_rnn_cell_10/MatMulMatMul4sequential_19/simple_rnn_10/strided_slice_2:output:0Lsequential_19/simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€–
Esequential_19/simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpNsequential_19_simple_rnn_10_simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Г
6sequential_19/simple_rnn_10/simple_rnn_cell_10/BiasAddBiasAdd?sequential_19/simple_rnn_10/simple_rnn_cell_10/MatMul:product:0Msequential_19/simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€÷
Fsequential_19/simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpOsequential_19_simple_rnn_10_simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0п
7sequential_19/simple_rnn_10/simple_rnn_cell_10/MatMul_1MatMul*sequential_19/simple_rnn_10/zeros:output:0Nsequential_19/simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€с
2sequential_19/simple_rnn_10/simple_rnn_cell_10/addAddV2?sequential_19/simple_rnn_10/simple_rnn_cell_10/BiasAdd:output:0Asequential_19/simple_rnn_10/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€•
3sequential_19/simple_rnn_10/simple_rnn_cell_10/ReluRelu6sequential_19/simple_rnn_10/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
9sequential_19/simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   М
+sequential_19/simple_rnn_10/TensorArrayV2_1TensorListReserveBsequential_19/simple_rnn_10/TensorArrayV2_1/element_shape:output:04sequential_19/simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“b
 sequential_19/simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : 
4sequential_19/simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€p
.sequential_19/simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : …
!sequential_19/simple_rnn_10/whileWhile7sequential_19/simple_rnn_10/while/loop_counter:output:0=sequential_19/simple_rnn_10/while/maximum_iterations:output:0)sequential_19/simple_rnn_10/time:output:04sequential_19/simple_rnn_10/TensorArrayV2_1:handle:0*sequential_19/simple_rnn_10/zeros:output:04sequential_19/simple_rnn_10/strided_slice_1:output:0Ssequential_19/simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_19_simple_rnn_10_simple_rnn_cell_10_matmul_readvariableop_resourceNsequential_19_simple_rnn_10_simple_rnn_cell_10_biasadd_readvariableop_resourceOsequential_19_simple_rnn_10_simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *:
body2R0
.sequential_19_simple_rnn_10_while_body_1734550*:
cond2R0
.sequential_19_simple_rnn_10_while_cond_1734549*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Э
Lsequential_19/simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ц
>sequential_19/simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack*sequential_19/simple_rnn_10/while:output:3Usequential_19/simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0Д
1sequential_19/simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€}
3sequential_19/simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3sequential_19/simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:У
+sequential_19/simple_rnn_10/strided_slice_3StridedSliceGsequential_19/simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0:sequential_19/simple_rnn_10/strided_slice_3/stack:output:0<sequential_19/simple_rnn_10/strided_slice_3/stack_1:output:0<sequential_19/simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskБ
,sequential_19/simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          к
'sequential_19/simple_rnn_10/transpose_1	TransposeGsequential_19/simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:05sequential_19/simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€|
!sequential_19/simple_rnn_11/ShapeShape+sequential_19/simple_rnn_10/transpose_1:y:0*
T0*
_output_shapes
:y
/sequential_19/simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_19/simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_19/simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
)sequential_19/simple_rnn_11/strided_sliceStridedSlice*sequential_19/simple_rnn_11/Shape:output:08sequential_19/simple_rnn_11/strided_slice/stack:output:0:sequential_19/simple_rnn_11/strided_slice/stack_1:output:0:sequential_19/simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*sequential_19/simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : «
(sequential_19/simple_rnn_11/zeros/packedPack2sequential_19/simple_rnn_11/strided_slice:output:03sequential_19/simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:l
'sequential_19/simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ј
!sequential_19/simple_rnn_11/zerosFill1sequential_19/simple_rnn_11/zeros/packed:output:00sequential_19/simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 
*sequential_19/simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"           
%sequential_19/simple_rnn_11/transpose	Transpose+sequential_19/simple_rnn_10/transpose_1:y:03sequential_19/simple_rnn_11/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€|
#sequential_19/simple_rnn_11/Shape_1Shape)sequential_19/simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:{
1sequential_19/simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_19/simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_19/simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+sequential_19/simple_rnn_11/strided_slice_1StridedSlice,sequential_19/simple_rnn_11/Shape_1:output:0:sequential_19/simple_rnn_11/strided_slice_1/stack:output:0<sequential_19/simple_rnn_11/strided_slice_1/stack_1:output:0<sequential_19/simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskВ
7sequential_19/simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€И
)sequential_19/simple_rnn_11/TensorArrayV2TensorListReserve@sequential_19/simple_rnn_11/TensorArrayV2/element_shape:output:04sequential_19/simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ґ
Qsequential_19/simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   і
Csequential_19/simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)sequential_19/simple_rnn_11/transpose:y:0Zsequential_19/simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“{
1sequential_19/simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_19/simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_19/simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+sequential_19/simple_rnn_11/strided_slice_2StridedSlice)sequential_19/simple_rnn_11/transpose:y:0:sequential_19/simple_rnn_11/strided_slice_2/stack:output:0<sequential_19/simple_rnn_11/strided_slice_2/stack_1:output:0<sequential_19/simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask“
Dsequential_19/simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpMsequential_19_simple_rnn_11_simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

: *
dtype0х
5sequential_19/simple_rnn_11/simple_rnn_cell_11/MatMulMatMul4sequential_19/simple_rnn_11/strided_slice_2:output:0Lsequential_19/simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ –
Esequential_19/simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpNsequential_19_simple_rnn_11_simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Г
6sequential_19/simple_rnn_11/simple_rnn_cell_11/BiasAddBiasAdd?sequential_19/simple_rnn_11/simple_rnn_cell_11/MatMul:product:0Msequential_19/simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ÷
Fsequential_19/simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpOsequential_19_simple_rnn_11_simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0п
7sequential_19/simple_rnn_11/simple_rnn_cell_11/MatMul_1MatMul*sequential_19/simple_rnn_11/zeros:output:0Nsequential_19/simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ с
2sequential_19/simple_rnn_11/simple_rnn_cell_11/addAddV2?sequential_19/simple_rnn_11/simple_rnn_cell_11/BiasAdd:output:0Asequential_19/simple_rnn_11/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ •
3sequential_19/simple_rnn_11/simple_rnn_cell_11/ReluRelu6sequential_19/simple_rnn_11/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ К
9sequential_19/simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    М
+sequential_19/simple_rnn_11/TensorArrayV2_1TensorListReserveBsequential_19/simple_rnn_11/TensorArrayV2_1/element_shape:output:04sequential_19/simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“b
 sequential_19/simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 
4sequential_19/simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€p
.sequential_19/simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : …
!sequential_19/simple_rnn_11/whileWhile7sequential_19/simple_rnn_11/while/loop_counter:output:0=sequential_19/simple_rnn_11/while/maximum_iterations:output:0)sequential_19/simple_rnn_11/time:output:04sequential_19/simple_rnn_11/TensorArrayV2_1:handle:0*sequential_19/simple_rnn_11/zeros:output:04sequential_19/simple_rnn_11/strided_slice_1:output:0Ssequential_19/simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_19_simple_rnn_11_simple_rnn_cell_11_matmul_readvariableop_resourceNsequential_19_simple_rnn_11_simple_rnn_cell_11_biasadd_readvariableop_resourceOsequential_19_simple_rnn_11_simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *:
body2R0
.sequential_19_simple_rnn_11_while_body_1734654*:
cond2R0
.sequential_19_simple_rnn_11_while_cond_1734653*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Э
Lsequential_19/simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ц
>sequential_19/simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack*sequential_19/simple_rnn_11/while:output:3Usequential_19/simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype0Д
1sequential_19/simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€}
3sequential_19/simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3sequential_19/simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:У
+sequential_19/simple_rnn_11/strided_slice_3StridedSliceGsequential_19/simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0:sequential_19/simple_rnn_11/strided_slice_3/stack:output:0<sequential_19/simple_rnn_11/strided_slice_3/stack_1:output:0<sequential_19/simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maskБ
,sequential_19/simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          к
'sequential_19/simple_rnn_11/transpose_1	TransposeGsequential_19/simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:05sequential_19/simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ А
/sequential_19/time_distributed_18/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ќ
)sequential_19/time_distributed_18/ReshapeReshape+sequential_19/simple_rnn_11/transpose_1:y:08sequential_19/time_distributed_18/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€  
@sequential_19/time_distributed_18/dense_21/MatMul/ReadVariableOpReadVariableOpIsequential_19_time_distributed_18_dense_21_matmul_readvariableop_resource*
_output_shapes

: *
dtype0л
1sequential_19/time_distributed_18/dense_21/MatMulMatMul2sequential_19/time_distributed_18/Reshape:output:0Hsequential_19/time_distributed_18/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€»
Asequential_19/time_distributed_18/dense_21/BiasAdd/ReadVariableOpReadVariableOpJsequential_19_time_distributed_18_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ч
2sequential_19/time_distributed_18/dense_21/BiasAddBiasAdd;sequential_19/time_distributed_18/dense_21/MatMul:product:0Isequential_19/time_distributed_18/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
1sequential_19/time_distributed_18/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€      е
+sequential_19/time_distributed_18/Reshape_1Reshape;sequential_19/time_distributed_18/dense_21/BiasAdd:output:0:sequential_19/time_distributed_18/Reshape_1/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€В
1sequential_19/time_distributed_18/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    —
+sequential_19/time_distributed_18/Reshape_2Reshape+sequential_19/simple_rnn_11/transpose_1:y:0:sequential_19/time_distributed_18/Reshape_2/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ З
IdentityIdentity4sequential_19/time_distributed_18/Reshape_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ѓ	
NoOpNoOpF^sequential_19/simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOpE^sequential_19/simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOpG^sequential_19/simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOp"^sequential_19/simple_rnn_10/whileF^sequential_19/simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOpE^sequential_19/simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOpG^sequential_19/simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOp"^sequential_19/simple_rnn_11/whileD^sequential_19/simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOpC^sequential_19/simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOpE^sequential_19/simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOp!^sequential_19/simple_rnn_8/whileD^sequential_19/simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOpC^sequential_19/simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOpE^sequential_19/simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOp!^sequential_19/simple_rnn_9/whileB^sequential_19/time_distributed_18/dense_21/BiasAdd/ReadVariableOpA^sequential_19/time_distributed_18/dense_21/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 2О
Esequential_19/simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOpEsequential_19/simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOp2М
Dsequential_19/simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOpDsequential_19/simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOp2Р
Fsequential_19/simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOpFsequential_19/simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOp2F
!sequential_19/simple_rnn_10/while!sequential_19/simple_rnn_10/while2О
Esequential_19/simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOpEsequential_19/simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOp2М
Dsequential_19/simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOpDsequential_19/simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOp2Р
Fsequential_19/simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOpFsequential_19/simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOp2F
!sequential_19/simple_rnn_11/while!sequential_19/simple_rnn_11/while2К
Csequential_19/simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOpCsequential_19/simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOp2И
Bsequential_19/simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOpBsequential_19/simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOp2М
Dsequential_19/simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOpDsequential_19/simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOp2D
 sequential_19/simple_rnn_8/while sequential_19/simple_rnn_8/while2К
Csequential_19/simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOpCsequential_19/simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOp2И
Bsequential_19/simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOpBsequential_19/simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOp2М
Dsequential_19/simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOpDsequential_19/simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOp2D
 sequential_19/simple_rnn_9/while sequential_19/simple_rnn_9/while2Ж
Asequential_19/time_distributed_18/dense_21/BiasAdd/ReadVariableOpAsequential_19/time_distributed_18/dense_21/BiasAdd/ReadVariableOp2Д
@sequential_19/time_distributed_18/dense_21/MatMul/ReadVariableOp@sequential_19/time_distributed_18/dense_21/MatMul/ReadVariableOp:_ [
+
_output_shapes
:€€€€€€€€€
,
_user_specified_namesimple_rnn_8_input
Ќµ
≈
J__inference_sequential_19_layer_call_and_return_conditional_losses_1738233

inputsO
=simple_rnn_8_simple_rnn_cell_8_matmul_readvariableop_resource: L
>simple_rnn_8_simple_rnn_cell_8_biasadd_readvariableop_resource: Q
?simple_rnn_8_simple_rnn_cell_8_matmul_1_readvariableop_resource:  O
=simple_rnn_9_simple_rnn_cell_9_matmul_readvariableop_resource: L
>simple_rnn_9_simple_rnn_cell_9_biasadd_readvariableop_resource:Q
?simple_rnn_9_simple_rnn_cell_9_matmul_1_readvariableop_resource:Q
?simple_rnn_10_simple_rnn_cell_10_matmul_readvariableop_resource:N
@simple_rnn_10_simple_rnn_cell_10_biasadd_readvariableop_resource:S
Asimple_rnn_10_simple_rnn_cell_10_matmul_1_readvariableop_resource:Q
?simple_rnn_11_simple_rnn_cell_11_matmul_readvariableop_resource: N
@simple_rnn_11_simple_rnn_cell_11_biasadd_readvariableop_resource: S
Asimple_rnn_11_simple_rnn_cell_11_matmul_1_readvariableop_resource:  M
;time_distributed_18_dense_21_matmul_readvariableop_resource: J
<time_distributed_18_dense_21_biasadd_readvariableop_resource:
identityИҐ7simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOpҐ6simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOpҐ8simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOpҐsimple_rnn_10/whileҐ7simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOpҐ6simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOpҐ8simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOpҐsimple_rnn_11/whileҐ5simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOpҐ4simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOpҐ6simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOpҐsimple_rnn_8/whileҐ5simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOpҐ4simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOpҐ6simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOpҐsimple_rnn_9/whileҐ3time_distributed_18/dense_21/BiasAdd/ReadVariableOpҐ2time_distributed_18/dense_21/MatMul/ReadVariableOpH
simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
simple_rnn_8/strided_sliceStridedSlicesimple_rnn_8/Shape:output:0)simple_rnn_8/strided_slice/stack:output:0+simple_rnn_8/strided_slice/stack_1:output:0+simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : Ъ
simple_rnn_8/zeros/packedPack#simple_rnn_8/strided_slice:output:0$simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    У
simple_rnn_8/zerosFill"simple_rnn_8/zeros/packed:output:0!simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ p
simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          З
simple_rnn_8/transpose	Transposeinputs$simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€^
simple_rnn_8/Shape_1Shapesimple_rnn_8/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
simple_rnn_8/strided_slice_1StridedSlicesimple_rnn_8/Shape_1:output:0+simple_rnn_8/strided_slice_1/stack:output:0-simple_rnn_8/strided_slice_1/stack_1:output:0-simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€џ
simple_rnn_8/TensorArrayV2TensorListReserve1simple_rnn_8/TensorArrayV2/element_shape:output:0%simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“У
Bsimple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   З
4simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_8/transpose:y:0Ksimple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“l
"simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:™
simple_rnn_8/strided_slice_2StridedSlicesimple_rnn_8/transpose:y:0+simple_rnn_8/strided_slice_2/stack:output:0-simple_rnn_8/strided_slice_2/stack_1:output:0-simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask≤
4simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp=simple_rnn_8_simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

: *
dtype0∆
%simple_rnn_8/simple_rnn_cell_8/MatMulMatMul%simple_rnn_8/strided_slice_2:output:0<simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ∞
5simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_8_simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0”
&simple_rnn_8/simple_rnn_cell_8/BiasAddBiasAdd/simple_rnn_8/simple_rnn_cell_8/MatMul:product:0=simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ґ
6simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_8_simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0ј
'simple_rnn_8/simple_rnn_cell_8/MatMul_1MatMulsimple_rnn_8/zeros:output:0>simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ѕ
"simple_rnn_8/simple_rnn_cell_8/addAddV2/simple_rnn_8/simple_rnn_cell_8/BiasAdd:output:01simple_rnn_8/simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ Е
#simple_rnn_8/simple_rnn_cell_8/ReluRelu&simple_rnn_8/simple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ {
*simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    я
simple_rnn_8/TensorArrayV2_1TensorListReserve3simple_rnn_8/TensorArrayV2_1/element_shape:output:0%simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“S
simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€a
simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Г
simple_rnn_8/whileWhile(simple_rnn_8/while/loop_counter:output:0.simple_rnn_8/while/maximum_iterations:output:0simple_rnn_8/time:output:0%simple_rnn_8/TensorArrayV2_1:handle:0simple_rnn_8/zeros:output:0%simple_rnn_8/strided_slice_1:output:0Dsimple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_8_simple_rnn_cell_8_matmul_readvariableop_resource>simple_rnn_8_simple_rnn_cell_8_biasadd_readvariableop_resource?simple_rnn_8_simple_rnn_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
simple_rnn_8_while_body_1737837*+
cond#R!
simple_rnn_8_while_cond_1737836*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations О
=simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    й
/simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_8/while:output:3Fsimple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype0u
"simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€n
$simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
simple_rnn_8/strided_slice_3StridedSlice8simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_8/strided_slice_3/stack:output:0-simple_rnn_8/strided_slice_3/stack_1:output:0-simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maskr
simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          љ
simple_rnn_8/transpose_1	Transpose8simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ ^
simple_rnn_9/ShapeShapesimple_rnn_8/transpose_1:y:0*
T0*
_output_shapes
:j
 simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
simple_rnn_9/strided_sliceStridedSlicesimple_rnn_9/Shape:output:0)simple_rnn_9/strided_slice/stack:output:0+simple_rnn_9/strided_slice/stack_1:output:0+simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ъ
simple_rnn_9/zeros/packedPack#simple_rnn_9/strided_slice:output:0$simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    У
simple_rnn_9/zerosFill"simple_rnn_9/zeros/packed:output:0!simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€p
simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Э
simple_rnn_9/transpose	Transposesimple_rnn_8/transpose_1:y:0$simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ ^
simple_rnn_9/Shape_1Shapesimple_rnn_9/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
simple_rnn_9/strided_slice_1StridedSlicesimple_rnn_9/Shape_1:output:0+simple_rnn_9/strided_slice_1/stack:output:0-simple_rnn_9/strided_slice_1/stack_1:output:0-simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€џ
simple_rnn_9/TensorArrayV2TensorListReserve1simple_rnn_9/TensorArrayV2/element_shape:output:0%simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“У
Bsimple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    З
4simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_9/transpose:y:0Ksimple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“l
"simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:™
simple_rnn_9/strided_slice_2StridedSlicesimple_rnn_9/transpose:y:0+simple_rnn_9/strided_slice_2/stack:output:0-simple_rnn_9/strided_slice_2/stack_1:output:0-simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask≤
4simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp=simple_rnn_9_simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype0∆
%simple_rnn_9/simple_rnn_cell_9/MatMulMatMul%simple_rnn_9/strided_slice_2:output:0<simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€∞
5simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_9_simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0”
&simple_rnn_9/simple_rnn_cell_9/BiasAddBiasAdd/simple_rnn_9/simple_rnn_cell_9/MatMul:product:0=simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ґ
6simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_9_simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0ј
'simple_rnn_9/simple_rnn_cell_9/MatMul_1MatMulsimple_rnn_9/zeros:output:0>simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ѕ
"simple_rnn_9/simple_rnn_cell_9/addAddV2/simple_rnn_9/simple_rnn_cell_9/BiasAdd:output:01simple_rnn_9/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Е
#simple_rnn_9/simple_rnn_cell_9/ReluRelu&simple_rnn_9/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€{
*simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   k
)simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :м
simple_rnn_9/TensorArrayV2_1TensorListReserve3simple_rnn_9/TensorArrayV2_1/element_shape:output:02simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“S
simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€a
simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Г
simple_rnn_9/whileWhile(simple_rnn_9/while/loop_counter:output:0.simple_rnn_9/while/maximum_iterations:output:0simple_rnn_9/time:output:0%simple_rnn_9/TensorArrayV2_1:handle:0simple_rnn_9/zeros:output:0%simple_rnn_9/strided_slice_1:output:0Dsimple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_9_simple_rnn_cell_9_matmul_readvariableop_resource>simple_rnn_9_simple_rnn_cell_9_biasadd_readvariableop_resource?simple_rnn_9_simple_rnn_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
simple_rnn_9_while_body_1737942*+
cond#R!
simple_rnn_9_while_cond_1737941*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations О
=simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   э
/simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_9/while:output:3Fsimple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0*
num_elementsu
"simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€n
$simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
simple_rnn_9/strided_slice_3StridedSlice8simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_9/strided_slice_3/stack:output:0-simple_rnn_9/strided_slice_3/stack_1:output:0-simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskr
simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          љ
simple_rnn_9/transpose_1	Transpose8simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€a
repeat_vector_18/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :∞
repeat_vector_18/ExpandDims
ExpandDims%simple_rnn_9/strided_slice_3:output:0(repeat_vector_18/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€k
repeat_vector_18/stackConst*
_output_shapes
:*
dtype0*!
valueB"         Ъ
repeat_vector_18/TileTile$repeat_vector_18/ExpandDims:output:0repeat_vector_18/stack:output:0*
T0*+
_output_shapes
:€€€€€€€€€a
simple_rnn_10/ShapeShaperepeat_vector_18/Tile:output:0*
T0*
_output_shapes
:k
!simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
simple_rnn_10/strided_sliceStridedSlicesimple_rnn_10/Shape:output:0*simple_rnn_10/strided_slice/stack:output:0,simple_rnn_10/strided_slice/stack_1:output:0,simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Э
simple_rnn_10/zeros/packedPack$simple_rnn_10/strided_slice:output:0%simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
simple_rnn_10/zerosFill#simple_rnn_10/zeros/packed:output:0"simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          °
simple_rnn_10/transpose	Transposerepeat_vector_18/Tile:output:0%simple_rnn_10/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€`
simple_rnn_10/Shape_1Shapesimple_rnn_10/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
simple_rnn_10/strided_slice_1StridedSlicesimple_rnn_10/Shape_1:output:0,simple_rnn_10/strided_slice_1/stack:output:0.simple_rnn_10/strided_slice_1/stack_1:output:0.simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ё
simple_rnn_10/TensorArrayV2TensorListReserve2simple_rnn_10/TensorArrayV2/element_shape:output:0&simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ф
Csimple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   К
5simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_10/transpose:y:0Lsimple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“m
#simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѓ
simple_rnn_10/strided_slice_2StridedSlicesimple_rnn_10/transpose:y:0,simple_rnn_10/strided_slice_2/stack:output:0.simple_rnn_10/strided_slice_2/stack_1:output:0.simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskґ
6simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp?simple_rnn_10_simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ћ
'simple_rnn_10/simple_rnn_cell_10/MatMulMatMul&simple_rnn_10/strided_slice_2:output:0>simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€і
7simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_10_simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ў
(simple_rnn_10/simple_rnn_cell_10/BiasAddBiasAdd1simple_rnn_10/simple_rnn_cell_10/MatMul:product:0?simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ї
8simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_10_simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0≈
)simple_rnn_10/simple_rnn_cell_10/MatMul_1MatMulsimple_rnn_10/zeros:output:0@simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€«
$simple_rnn_10/simple_rnn_cell_10/addAddV21simple_rnn_10/simple_rnn_cell_10/BiasAdd:output:03simple_rnn_10/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Й
%simple_rnn_10/simple_rnn_cell_10/ReluRelu(simple_rnn_10/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
+simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   в
simple_rnn_10/TensorArrayV2_1TensorListReserve4simple_rnn_10/TensorArrayV2_1/element_shape:output:0&simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“T
simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€b
 simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : У
simple_rnn_10/whileWhile)simple_rnn_10/while/loop_counter:output:0/simple_rnn_10/while/maximum_iterations:output:0simple_rnn_10/time:output:0&simple_rnn_10/TensorArrayV2_1:handle:0simple_rnn_10/zeros:output:0&simple_rnn_10/strided_slice_1:output:0Esimple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_10_simple_rnn_cell_10_matmul_readvariableop_resource@simple_rnn_10_simple_rnn_cell_10_biasadd_readvariableop_resourceAsimple_rnn_10_simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_10_while_body_1738051*,
cond$R"
 simple_rnn_10_while_cond_1738050*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations П
>simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   м
0simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_10/while:output:3Gsimple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0v
#simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€o
%simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ќ
simple_rnn_10/strided_slice_3StridedSlice9simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_10/strided_slice_3/stack:output:0.simple_rnn_10/strided_slice_3/stack_1:output:0.simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_masks
simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ј
simple_rnn_10/transpose_1	Transpose9simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€`
simple_rnn_11/ShapeShapesimple_rnn_10/transpose_1:y:0*
T0*
_output_shapes
:k
!simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
simple_rnn_11/strided_sliceStridedSlicesimple_rnn_11/Shape:output:0*simple_rnn_11/strided_slice/stack:output:0,simple_rnn_11/strided_slice/stack_1:output:0,simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : Э
simple_rnn_11/zeros/packedPack$simple_rnn_11/strided_slice:output:0%simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
simple_rnn_11/zerosFill#simple_rnn_11/zeros/packed:output:0"simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ q
simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
simple_rnn_11/transpose	Transposesimple_rnn_10/transpose_1:y:0%simple_rnn_11/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€`
simple_rnn_11/Shape_1Shapesimple_rnn_11/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
simple_rnn_11/strided_slice_1StridedSlicesimple_rnn_11/Shape_1:output:0,simple_rnn_11/strided_slice_1/stack:output:0.simple_rnn_11/strided_slice_1/stack_1:output:0.simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ё
simple_rnn_11/TensorArrayV2TensorListReserve2simple_rnn_11/TensorArrayV2/element_shape:output:0&simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ф
Csimple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   К
5simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_11/transpose:y:0Lsimple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“m
#simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѓ
simple_rnn_11/strided_slice_2StridedSlicesimple_rnn_11/transpose:y:0,simple_rnn_11/strided_slice_2/stack:output:0.simple_rnn_11/strided_slice_2/stack_1:output:0.simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskґ
6simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp?simple_rnn_11_simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ћ
'simple_rnn_11/simple_rnn_cell_11/MatMulMatMul&simple_rnn_11/strided_slice_2:output:0>simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ і
7simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_11_simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ў
(simple_rnn_11/simple_rnn_cell_11/BiasAddBiasAdd1simple_rnn_11/simple_rnn_cell_11/MatMul:product:0?simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ї
8simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_11_simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0≈
)simple_rnn_11/simple_rnn_cell_11/MatMul_1MatMulsimple_rnn_11/zeros:output:0@simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ «
$simple_rnn_11/simple_rnn_cell_11/addAddV21simple_rnn_11/simple_rnn_cell_11/BiasAdd:output:03simple_rnn_11/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ Й
%simple_rnn_11/simple_rnn_cell_11/ReluRelu(simple_rnn_11/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ |
+simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    в
simple_rnn_11/TensorArrayV2_1TensorListReserve4simple_rnn_11/TensorArrayV2_1/element_shape:output:0&simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“T
simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€b
 simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : У
simple_rnn_11/whileWhile)simple_rnn_11/while/loop_counter:output:0/simple_rnn_11/while/maximum_iterations:output:0simple_rnn_11/time:output:0&simple_rnn_11/TensorArrayV2_1:handle:0simple_rnn_11/zeros:output:0&simple_rnn_11/strided_slice_1:output:0Esimple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_11_simple_rnn_cell_11_matmul_readvariableop_resource@simple_rnn_11_simple_rnn_cell_11_biasadd_readvariableop_resourceAsimple_rnn_11_simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_11_while_body_1738155*,
cond$R"
 simple_rnn_11_while_cond_1738154*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations П
>simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    м
0simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_11/while:output:3Gsimple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype0v
#simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€o
%simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ќ
simple_rnn_11/strided_slice_3StridedSlice9simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_11/strided_slice_3/stack:output:0.simple_rnn_11/strided_slice_3/stack_1:output:0.simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_masks
simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ј
simple_rnn_11/transpose_1	Transpose9simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ r
!time_distributed_18/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    £
time_distributed_18/ReshapeReshapesimple_rnn_11/transpose_1:y:0*time_distributed_18/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ѓ
2time_distributed_18/dense_21/MatMul/ReadVariableOpReadVariableOp;time_distributed_18_dense_21_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ѕ
#time_distributed_18/dense_21/MatMulMatMul$time_distributed_18/Reshape:output:0:time_distributed_18/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
3time_distributed_18/dense_21/BiasAdd/ReadVariableOpReadVariableOp<time_distributed_18_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
$time_distributed_18/dense_21/BiasAddBiasAdd-time_distributed_18/dense_21/MatMul:product:0;time_distributed_18/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
#time_distributed_18/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€      ї
time_distributed_18/Reshape_1Reshape-time_distributed_18/dense_21/BiasAdd:output:0,time_distributed_18/Reshape_1/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€t
#time_distributed_18/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    І
time_distributed_18/Reshape_2Reshapesimple_rnn_11/transpose_1:y:0,time_distributed_18/Reshape_2/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ y
IdentityIdentity&time_distributed_18/Reshape_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€≥
NoOpNoOp8^simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOp7^simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOp9^simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOp^simple_rnn_10/while8^simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOp7^simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOp9^simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOp^simple_rnn_11/while6^simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOp5^simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOp7^simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOp^simple_rnn_8/while6^simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOp5^simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOp7^simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOp^simple_rnn_9/while4^time_distributed_18/dense_21/BiasAdd/ReadVariableOp3^time_distributed_18/dense_21/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 2r
7simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOp7simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOp2p
6simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOp6simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOp2t
8simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOp8simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOp2*
simple_rnn_10/whilesimple_rnn_10/while2r
7simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOp7simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOp2p
6simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOp6simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOp2t
8simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOp8simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOp2*
simple_rnn_11/whilesimple_rnn_11/while2n
5simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOp5simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOp2l
4simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOp4simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOp2p
6simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOp6simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOp2(
simple_rnn_8/whilesimple_rnn_8/while2n
5simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOp5simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOp2l
4simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOp4simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOp2p
6simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOp6simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOp2(
simple_rnn_9/whilesimple_rnn_9/while2j
3time_distributed_18/dense_21/BiasAdd/ReadVariableOp3time_distributed_18/dense_21/BiasAdd/ReadVariableOp2h
2time_distributed_18/dense_21/MatMul/ReadVariableOp2time_distributed_18/dense_21/MatMul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
х%
е
J__inference_sequential_19_layer_call_and_return_conditional_losses_1737106

inputs&
simple_rnn_8_1737069: "
simple_rnn_8_1737071: &
simple_rnn_8_1737073:  &
simple_rnn_9_1737076: "
simple_rnn_9_1737078:&
simple_rnn_9_1737080:'
simple_rnn_10_1737084:#
simple_rnn_10_1737086:'
simple_rnn_10_1737088:'
simple_rnn_11_1737091: #
simple_rnn_11_1737093: '
simple_rnn_11_1737095:  -
time_distributed_18_1737098: )
time_distributed_18_1737100:
identityИҐ%simple_rnn_10/StatefulPartitionedCallҐ%simple_rnn_11/StatefulPartitionedCallҐ$simple_rnn_8/StatefulPartitionedCallҐ$simple_rnn_9/StatefulPartitionedCallҐ+time_distributed_18/StatefulPartitionedCallЯ
$simple_rnn_8/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_8_1737069simple_rnn_8_1737071simple_rnn_8_1737073*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1737022¬
$simple_rnn_9/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_8/StatefulPartitionedCall:output:0simple_rnn_9_1737076simple_rnn_9_1737078simple_rnn_9_1737080*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1736892ф
 repeat_vector_18/PartitionedCallPartitionedCall-simple_rnn_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_repeat_vector_18_layer_call_and_return_conditional_losses_1735332«
%simple_rnn_10/StatefulPartitionedCallStatefulPartitionedCall)repeat_vector_18/PartitionedCall:output:0simple_rnn_10_1737084simple_rnn_10_1737086simple_rnn_10_1737088*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1736760ћ
%simple_rnn_11/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_10/StatefulPartitionedCall:output:0simple_rnn_11_1737091simple_rnn_11_1737093simple_rnn_11_1737095*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1736630Ћ
+time_distributed_18/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_11/StatefulPartitionedCall:output:0time_distributed_18_1737098time_distributed_18_1737100*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_time_distributed_18_layer_call_and_return_conditional_losses_1735993r
!time_distributed_18/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    і
time_distributed_18/ReshapeReshape.simple_rnn_11/StatefulPartitionedCall:output:0*time_distributed_18/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ З
IdentityIdentity4time_distributed_18/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Т
NoOpNoOp&^simple_rnn_10/StatefulPartitionedCall&^simple_rnn_11/StatefulPartitionedCall%^simple_rnn_8/StatefulPartitionedCall%^simple_rnn_9/StatefulPartitionedCall,^time_distributed_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 2N
%simple_rnn_10/StatefulPartitionedCall%simple_rnn_10/StatefulPartitionedCall2N
%simple_rnn_11/StatefulPartitionedCall%simple_rnn_11/StatefulPartitionedCall2L
$simple_rnn_8/StatefulPartitionedCall$simple_rnn_8/StatefulPartitionedCall2L
$simple_rnn_9/StatefulPartitionedCall$simple_rnn_9/StatefulPartitionedCall2Z
+time_distributed_18/StatefulPartitionedCall+time_distributed_18/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
м,
“
while_body_1739508
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:ИҐ/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_10/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0®
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0≈
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€¶
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ѕ
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0ђ
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ѓ
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€y
while/simple_rnn_cell_10/ReluRelu while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€‘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder+while/simple_rnn_cell_10/Relu:activations:0*
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
: И
while/Identity_4Identity+while/simple_rnn_cell_10/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€в

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
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2b
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
Џ

є
 simple_rnn_11_while_cond_17381548
4simple_rnn_11_while_simple_rnn_11_while_loop_counter>
:simple_rnn_11_while_simple_rnn_11_while_maximum_iterations#
simple_rnn_11_while_placeholder%
!simple_rnn_11_while_placeholder_1%
!simple_rnn_11_while_placeholder_2:
6simple_rnn_11_while_less_simple_rnn_11_strided_slice_1Q
Msimple_rnn_11_while_simple_rnn_11_while_cond_1738154___redundant_placeholder0Q
Msimple_rnn_11_while_simple_rnn_11_while_cond_1738154___redundant_placeholder1Q
Msimple_rnn_11_while_simple_rnn_11_while_cond_1738154___redundant_placeholder2Q
Msimple_rnn_11_while_simple_rnn_11_while_cond_1738154___redundant_placeholder3 
simple_rnn_11_while_identity
Ъ
simple_rnn_11/while/LessLesssimple_rnn_11_while_placeholder6simple_rnn_11_while_less_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_11/while/IdentityIdentitysimple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_11_while_identity%simple_rnn_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Щ
м
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_1740387

inputs
states_00
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€G
ReluReluadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states/0
я
ѓ
while_cond_1738318
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1738318___redundant_placeholder05
1while_while_cond_1738318___redundant_placeholder15
1while_while_cond_1738318___redundant_placeholder25
1while_while_cond_1738318___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
ј,
…
while_body_1736956
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0: L
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_8_matmul_readvariableop_resource: E
7while_simple_rnn_cell_8_biasadd_readvariableop_resource: J
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:  ИҐ.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_8/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0¶
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0√
while/simple_rnn_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ §
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Њ
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ™
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0™
 while/simple_rnn_cell_8/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ђ
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ w
while/simple_rnn_cell_8/ReluReluwhile/simple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ”
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_8/Relu:activations:0*
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
: З
while/Identity_4Identity*while/simple_rnn_cell_8/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ я

while/NoOpNoOp/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2`
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_8/MatMul/ReadVariableOp-while/simple_rnn_cell_8/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
 =
√
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1739574

inputsC
1simple_rnn_cell_10_matmul_readvariableop_resource:@
2simple_rnn_cell_10_biasadd_readvariableop_resource:E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:
identityИҐ)simple_rnn_cell_10/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_10/MatMul/ReadVariableOpҐ*simple_rnn_cell_10/MatMul_1/ReadVariableOpҐwhile;
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
value	B :s
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
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
valueB"€€€€   а
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
:€€€€€€€€€*
shrink_axis_maskЪ
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype0°
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Ы
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€m
simple_rnn_cell_10/ReluRelusimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1739508*
condR
while_cond_1739507*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0h
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
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€“
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
‘8
ѕ
simple_rnn_8_while_body_17378376
2simple_rnn_8_while_simple_rnn_8_while_loop_counter<
8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations"
simple_rnn_8_while_placeholder$
 simple_rnn_8_while_placeholder_1$
 simple_rnn_8_while_placeholder_25
1simple_rnn_8_while_simple_rnn_8_strided_slice_1_0q
msimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_8_while_simple_rnn_cell_8_matmul_readvariableop_resource_0: T
Fsimple_rnn_8_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0: Y
Gsimple_rnn_8_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0:  
simple_rnn_8_while_identity!
simple_rnn_8_while_identity_1!
simple_rnn_8_while_identity_2!
simple_rnn_8_while_identity_3!
simple_rnn_8_while_identity_43
/simple_rnn_8_while_simple_rnn_8_strided_slice_1o
ksimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_8_while_simple_rnn_cell_8_matmul_readvariableop_resource: R
Dsimple_rnn_8_while_simple_rnn_cell_8_biasadd_readvariableop_resource: W
Esimple_rnn_8_while_simple_rnn_cell_8_matmul_1_readvariableop_resource:  ИҐ;simple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpҐ:simple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOpҐ<simple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpХ
Dsimple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   з
6simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_8_while_placeholderMsimple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0ј
:simple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_8_while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0к
+simple_rnn_8/while/simple_rnn_cell_8/MatMulMatMul=simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Њ
;simple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_8_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0е
,simple_rnn_8/while/simple_rnn_cell_8/BiasAddBiasAdd5simple_rnn_8/while/simple_rnn_cell_8/MatMul:product:0Csimple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ƒ
<simple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_8_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0—
-simple_rnn_8/while/simple_rnn_cell_8/MatMul_1MatMul simple_rnn_8_while_placeholder_2Dsimple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ”
(simple_rnn_8/while/simple_rnn_cell_8/addAddV25simple_rnn_8/while/simple_rnn_cell_8/BiasAdd:output:07simple_rnn_8/while/simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ С
)simple_rnn_8/while/simple_rnn_cell_8/ReluRelu,simple_rnn_8/while/simple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ З
7simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_8_while_placeholder_1simple_rnn_8_while_placeholder7simple_rnn_8/while/simple_rnn_cell_8/Relu:activations:0*
_output_shapes
: *
element_dtype0:йи“Z
simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Г
simple_rnn_8/while/addAddV2simple_rnn_8_while_placeholder!simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ы
simple_rnn_8/while/add_1AddV22simple_rnn_8_while_simple_rnn_8_while_loop_counter#simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: А
simple_rnn_8/while/IdentityIdentitysimple_rnn_8/while/add_1:z:0^simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Ю
simple_rnn_8/while/Identity_1Identity8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations^simple_rnn_8/while/NoOp*
T0*
_output_shapes
: А
simple_rnn_8/while/Identity_2Identitysimple_rnn_8/while/add:z:0^simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ≠
simple_rnn_8/while/Identity_3IdentityGsimple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Ѓ
simple_rnn_8/while/Identity_4Identity7simple_rnn_8/while/simple_rnn_cell_8/Relu:activations:0^simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ У
simple_rnn_8/while/NoOpNoOp<^simple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp;^simple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOp=^simple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_8_while_identity$simple_rnn_8/while/Identity:output:0"G
simple_rnn_8_while_identity_1&simple_rnn_8/while/Identity_1:output:0"G
simple_rnn_8_while_identity_2&simple_rnn_8/while/Identity_2:output:0"G
simple_rnn_8_while_identity_3&simple_rnn_8/while/Identity_3:output:0"G
simple_rnn_8_while_identity_4&simple_rnn_8/while/Identity_4:output:0"d
/simple_rnn_8_while_simple_rnn_8_strided_slice_11simple_rnn_8_while_simple_rnn_8_strided_slice_1_0"О
Dsimple_rnn_8_while_simple_rnn_cell_8_biasadd_readvariableop_resourceFsimple_rnn_8_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"Р
Esimple_rnn_8_while_simple_rnn_cell_8_matmul_1_readvariableop_resourceGsimple_rnn_8_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"М
Csimple_rnn_8_while_simple_rnn_cell_8_matmul_readvariableop_resourceEsimple_rnn_8_while_simple_rnn_cell_8_matmul_readvariableop_resource_0"№
ksimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensormsimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2z
;simple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp;simple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2x
:simple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOp:simple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOp2|
<simple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp<simple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Д
•
P__inference_time_distributed_18_layer_call_and_return_conditional_losses_1740218

inputs9
'dense_21_matmul_readvariableop_resource: 6
(dense_21_biasadd_readvariableop_resource:
identityИҐdense_21/BiasAdd/ReadVariableOpҐdense_21/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
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
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ж
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Е
dense_21/MatMulMatMulReshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:И
	Reshape_1Reshapedense_21/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€Й
NoOpNoOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€ : : 2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
÷
н
%__inference_signature_wrapper_1737291
simple_rnn_8_input
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8: 
	unknown_9: 

unknown_10:  

unknown_11: 

unknown_12:
identityИҐStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__wrapped_model_1734732s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:€€€€€€€€€
,
_user_specified_namesimple_rnn_8_input
№-
…
while_body_1736164
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_9_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_9_matmul_readvariableop_resource: E
7while_simple_rnn_cell_9_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:ИҐ.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_9/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0¶
-while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0√
while/simple_rnn_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€§
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Њ
while/simple_rnn_cell_9/BiasAddBiasAdd(while/simple_rnn_cell_9/MatMul:product:06while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€™
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0™
 while/simple_rnn_cell_9/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
while/simple_rnn_cell_9/addAddV2(while/simple_rnn_cell_9/BiasAdd:output:0*while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€w
while/simple_rnn_cell_9/ReluReluwhile/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ы
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_9/Relu:activations:0*
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
: З
while/Identity_4Identity*while/simple_rnn_cell_9/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€я

while/NoOpNoOp/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_9_biasadd_readvariableop_resource9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_9_matmul_readvariableop_resource8while_simple_rnn_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2`
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_9/MatMul/ReadVariableOp-while/simple_rnn_cell_9/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
Ќµ
≈
J__inference_sequential_19_layer_call_and_return_conditional_losses_1737795

inputsO
=simple_rnn_8_simple_rnn_cell_8_matmul_readvariableop_resource: L
>simple_rnn_8_simple_rnn_cell_8_biasadd_readvariableop_resource: Q
?simple_rnn_8_simple_rnn_cell_8_matmul_1_readvariableop_resource:  O
=simple_rnn_9_simple_rnn_cell_9_matmul_readvariableop_resource: L
>simple_rnn_9_simple_rnn_cell_9_biasadd_readvariableop_resource:Q
?simple_rnn_9_simple_rnn_cell_9_matmul_1_readvariableop_resource:Q
?simple_rnn_10_simple_rnn_cell_10_matmul_readvariableop_resource:N
@simple_rnn_10_simple_rnn_cell_10_biasadd_readvariableop_resource:S
Asimple_rnn_10_simple_rnn_cell_10_matmul_1_readvariableop_resource:Q
?simple_rnn_11_simple_rnn_cell_11_matmul_readvariableop_resource: N
@simple_rnn_11_simple_rnn_cell_11_biasadd_readvariableop_resource: S
Asimple_rnn_11_simple_rnn_cell_11_matmul_1_readvariableop_resource:  M
;time_distributed_18_dense_21_matmul_readvariableop_resource: J
<time_distributed_18_dense_21_biasadd_readvariableop_resource:
identityИҐ7simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOpҐ6simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOpҐ8simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOpҐsimple_rnn_10/whileҐ7simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOpҐ6simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOpҐ8simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOpҐsimple_rnn_11/whileҐ5simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOpҐ4simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOpҐ6simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOpҐsimple_rnn_8/whileҐ5simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOpҐ4simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOpҐ6simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOpҐsimple_rnn_9/whileҐ3time_distributed_18/dense_21/BiasAdd/ReadVariableOpҐ2time_distributed_18/dense_21/MatMul/ReadVariableOpH
simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
simple_rnn_8/strided_sliceStridedSlicesimple_rnn_8/Shape:output:0)simple_rnn_8/strided_slice/stack:output:0+simple_rnn_8/strided_slice/stack_1:output:0+simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : Ъ
simple_rnn_8/zeros/packedPack#simple_rnn_8/strided_slice:output:0$simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    У
simple_rnn_8/zerosFill"simple_rnn_8/zeros/packed:output:0!simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ p
simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          З
simple_rnn_8/transpose	Transposeinputs$simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€^
simple_rnn_8/Shape_1Shapesimple_rnn_8/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
simple_rnn_8/strided_slice_1StridedSlicesimple_rnn_8/Shape_1:output:0+simple_rnn_8/strided_slice_1/stack:output:0-simple_rnn_8/strided_slice_1/stack_1:output:0-simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€џ
simple_rnn_8/TensorArrayV2TensorListReserve1simple_rnn_8/TensorArrayV2/element_shape:output:0%simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“У
Bsimple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   З
4simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_8/transpose:y:0Ksimple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“l
"simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:™
simple_rnn_8/strided_slice_2StridedSlicesimple_rnn_8/transpose:y:0+simple_rnn_8/strided_slice_2/stack:output:0-simple_rnn_8/strided_slice_2/stack_1:output:0-simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask≤
4simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp=simple_rnn_8_simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

: *
dtype0∆
%simple_rnn_8/simple_rnn_cell_8/MatMulMatMul%simple_rnn_8/strided_slice_2:output:0<simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ∞
5simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_8_simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0”
&simple_rnn_8/simple_rnn_cell_8/BiasAddBiasAdd/simple_rnn_8/simple_rnn_cell_8/MatMul:product:0=simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ґ
6simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_8_simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0ј
'simple_rnn_8/simple_rnn_cell_8/MatMul_1MatMulsimple_rnn_8/zeros:output:0>simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ѕ
"simple_rnn_8/simple_rnn_cell_8/addAddV2/simple_rnn_8/simple_rnn_cell_8/BiasAdd:output:01simple_rnn_8/simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ Е
#simple_rnn_8/simple_rnn_cell_8/ReluRelu&simple_rnn_8/simple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ {
*simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    я
simple_rnn_8/TensorArrayV2_1TensorListReserve3simple_rnn_8/TensorArrayV2_1/element_shape:output:0%simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“S
simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€a
simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Г
simple_rnn_8/whileWhile(simple_rnn_8/while/loop_counter:output:0.simple_rnn_8/while/maximum_iterations:output:0simple_rnn_8/time:output:0%simple_rnn_8/TensorArrayV2_1:handle:0simple_rnn_8/zeros:output:0%simple_rnn_8/strided_slice_1:output:0Dsimple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_8_simple_rnn_cell_8_matmul_readvariableop_resource>simple_rnn_8_simple_rnn_cell_8_biasadd_readvariableop_resource?simple_rnn_8_simple_rnn_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
simple_rnn_8_while_body_1737399*+
cond#R!
simple_rnn_8_while_cond_1737398*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations О
=simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    й
/simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_8/while:output:3Fsimple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype0u
"simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€n
$simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
simple_rnn_8/strided_slice_3StridedSlice8simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_8/strided_slice_3/stack:output:0-simple_rnn_8/strided_slice_3/stack_1:output:0-simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maskr
simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          љ
simple_rnn_8/transpose_1	Transpose8simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ ^
simple_rnn_9/ShapeShapesimple_rnn_8/transpose_1:y:0*
T0*
_output_shapes
:j
 simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
simple_rnn_9/strided_sliceStridedSlicesimple_rnn_9/Shape:output:0)simple_rnn_9/strided_slice/stack:output:0+simple_rnn_9/strided_slice/stack_1:output:0+simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ъ
simple_rnn_9/zeros/packedPack#simple_rnn_9/strided_slice:output:0$simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    У
simple_rnn_9/zerosFill"simple_rnn_9/zeros/packed:output:0!simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€p
simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Э
simple_rnn_9/transpose	Transposesimple_rnn_8/transpose_1:y:0$simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ ^
simple_rnn_9/Shape_1Shapesimple_rnn_9/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
simple_rnn_9/strided_slice_1StridedSlicesimple_rnn_9/Shape_1:output:0+simple_rnn_9/strided_slice_1/stack:output:0-simple_rnn_9/strided_slice_1/stack_1:output:0-simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€џ
simple_rnn_9/TensorArrayV2TensorListReserve1simple_rnn_9/TensorArrayV2/element_shape:output:0%simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“У
Bsimple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    З
4simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_9/transpose:y:0Ksimple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“l
"simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:™
simple_rnn_9/strided_slice_2StridedSlicesimple_rnn_9/transpose:y:0+simple_rnn_9/strided_slice_2/stack:output:0-simple_rnn_9/strided_slice_2/stack_1:output:0-simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask≤
4simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp=simple_rnn_9_simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype0∆
%simple_rnn_9/simple_rnn_cell_9/MatMulMatMul%simple_rnn_9/strided_slice_2:output:0<simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€∞
5simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_9_simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0”
&simple_rnn_9/simple_rnn_cell_9/BiasAddBiasAdd/simple_rnn_9/simple_rnn_cell_9/MatMul:product:0=simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ґ
6simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_9_simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0ј
'simple_rnn_9/simple_rnn_cell_9/MatMul_1MatMulsimple_rnn_9/zeros:output:0>simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ѕ
"simple_rnn_9/simple_rnn_cell_9/addAddV2/simple_rnn_9/simple_rnn_cell_9/BiasAdd:output:01simple_rnn_9/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Е
#simple_rnn_9/simple_rnn_cell_9/ReluRelu&simple_rnn_9/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€{
*simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   k
)simple_rnn_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :м
simple_rnn_9/TensorArrayV2_1TensorListReserve3simple_rnn_9/TensorArrayV2_1/element_shape:output:02simple_rnn_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“S
simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€a
simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Г
simple_rnn_9/whileWhile(simple_rnn_9/while/loop_counter:output:0.simple_rnn_9/while/maximum_iterations:output:0simple_rnn_9/time:output:0%simple_rnn_9/TensorArrayV2_1:handle:0simple_rnn_9/zeros:output:0%simple_rnn_9/strided_slice_1:output:0Dsimple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_9_simple_rnn_cell_9_matmul_readvariableop_resource>simple_rnn_9_simple_rnn_cell_9_biasadd_readvariableop_resource?simple_rnn_9_simple_rnn_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
simple_rnn_9_while_body_1737504*+
cond#R!
simple_rnn_9_while_cond_1737503*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations О
=simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   э
/simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_9/while:output:3Fsimple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0*
num_elementsu
"simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€n
$simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
simple_rnn_9/strided_slice_3StridedSlice8simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_9/strided_slice_3/stack:output:0-simple_rnn_9/strided_slice_3/stack_1:output:0-simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskr
simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          љ
simple_rnn_9/transpose_1	Transpose8simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€a
repeat_vector_18/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :∞
repeat_vector_18/ExpandDims
ExpandDims%simple_rnn_9/strided_slice_3:output:0(repeat_vector_18/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€k
repeat_vector_18/stackConst*
_output_shapes
:*
dtype0*!
valueB"         Ъ
repeat_vector_18/TileTile$repeat_vector_18/ExpandDims:output:0repeat_vector_18/stack:output:0*
T0*+
_output_shapes
:€€€€€€€€€a
simple_rnn_10/ShapeShaperepeat_vector_18/Tile:output:0*
T0*
_output_shapes
:k
!simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
simple_rnn_10/strided_sliceStridedSlicesimple_rnn_10/Shape:output:0*simple_rnn_10/strided_slice/stack:output:0,simple_rnn_10/strided_slice/stack_1:output:0,simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Э
simple_rnn_10/zeros/packedPack$simple_rnn_10/strided_slice:output:0%simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
simple_rnn_10/zerosFill#simple_rnn_10/zeros/packed:output:0"simple_rnn_10/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          °
simple_rnn_10/transpose	Transposerepeat_vector_18/Tile:output:0%simple_rnn_10/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€`
simple_rnn_10/Shape_1Shapesimple_rnn_10/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
simple_rnn_10/strided_slice_1StridedSlicesimple_rnn_10/Shape_1:output:0,simple_rnn_10/strided_slice_1/stack:output:0.simple_rnn_10/strided_slice_1/stack_1:output:0.simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ё
simple_rnn_10/TensorArrayV2TensorListReserve2simple_rnn_10/TensorArrayV2/element_shape:output:0&simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ф
Csimple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   К
5simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_10/transpose:y:0Lsimple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“m
#simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѓ
simple_rnn_10/strided_slice_2StridedSlicesimple_rnn_10/transpose:y:0,simple_rnn_10/strided_slice_2/stack:output:0.simple_rnn_10/strided_slice_2/stack_1:output:0.simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskґ
6simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp?simple_rnn_10_simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ћ
'simple_rnn_10/simple_rnn_cell_10/MatMulMatMul&simple_rnn_10/strided_slice_2:output:0>simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€і
7simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_10_simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ў
(simple_rnn_10/simple_rnn_cell_10/BiasAddBiasAdd1simple_rnn_10/simple_rnn_cell_10/MatMul:product:0?simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ї
8simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_10_simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0≈
)simple_rnn_10/simple_rnn_cell_10/MatMul_1MatMulsimple_rnn_10/zeros:output:0@simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€«
$simple_rnn_10/simple_rnn_cell_10/addAddV21simple_rnn_10/simple_rnn_cell_10/BiasAdd:output:03simple_rnn_10/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Й
%simple_rnn_10/simple_rnn_cell_10/ReluRelu(simple_rnn_10/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
+simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   в
simple_rnn_10/TensorArrayV2_1TensorListReserve4simple_rnn_10/TensorArrayV2_1/element_shape:output:0&simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“T
simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€b
 simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : У
simple_rnn_10/whileWhile)simple_rnn_10/while/loop_counter:output:0/simple_rnn_10/while/maximum_iterations:output:0simple_rnn_10/time:output:0&simple_rnn_10/TensorArrayV2_1:handle:0simple_rnn_10/zeros:output:0&simple_rnn_10/strided_slice_1:output:0Esimple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_10_simple_rnn_cell_10_matmul_readvariableop_resource@simple_rnn_10_simple_rnn_cell_10_biasadd_readvariableop_resourceAsimple_rnn_10_simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_10_while_body_1737613*,
cond$R"
 simple_rnn_10_while_cond_1737612*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations П
>simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   м
0simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_10/while:output:3Gsimple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0v
#simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€o
%simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ќ
simple_rnn_10/strided_slice_3StridedSlice9simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_10/strided_slice_3/stack:output:0.simple_rnn_10/strided_slice_3/stack_1:output:0.simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_masks
simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ј
simple_rnn_10/transpose_1	Transpose9simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€`
simple_rnn_11/ShapeShapesimple_rnn_10/transpose_1:y:0*
T0*
_output_shapes
:k
!simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
simple_rnn_11/strided_sliceStridedSlicesimple_rnn_11/Shape:output:0*simple_rnn_11/strided_slice/stack:output:0,simple_rnn_11/strided_slice/stack_1:output:0,simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : Э
simple_rnn_11/zeros/packedPack$simple_rnn_11/strided_slice:output:0%simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
simple_rnn_11/zerosFill#simple_rnn_11/zeros/packed:output:0"simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ q
simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
simple_rnn_11/transpose	Transposesimple_rnn_10/transpose_1:y:0%simple_rnn_11/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€`
simple_rnn_11/Shape_1Shapesimple_rnn_11/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
simple_rnn_11/strided_slice_1StridedSlicesimple_rnn_11/Shape_1:output:0,simple_rnn_11/strided_slice_1/stack:output:0.simple_rnn_11/strided_slice_1/stack_1:output:0.simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ё
simple_rnn_11/TensorArrayV2TensorListReserve2simple_rnn_11/TensorArrayV2/element_shape:output:0&simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ф
Csimple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   К
5simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_11/transpose:y:0Lsimple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“m
#simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѓ
simple_rnn_11/strided_slice_2StridedSlicesimple_rnn_11/transpose:y:0,simple_rnn_11/strided_slice_2/stack:output:0.simple_rnn_11/strided_slice_2/stack_1:output:0.simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskґ
6simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp?simple_rnn_11_simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ћ
'simple_rnn_11/simple_rnn_cell_11/MatMulMatMul&simple_rnn_11/strided_slice_2:output:0>simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ і
7simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_11_simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ў
(simple_rnn_11/simple_rnn_cell_11/BiasAddBiasAdd1simple_rnn_11/simple_rnn_cell_11/MatMul:product:0?simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ї
8simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_11_simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0≈
)simple_rnn_11/simple_rnn_cell_11/MatMul_1MatMulsimple_rnn_11/zeros:output:0@simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ «
$simple_rnn_11/simple_rnn_cell_11/addAddV21simple_rnn_11/simple_rnn_cell_11/BiasAdd:output:03simple_rnn_11/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ Й
%simple_rnn_11/simple_rnn_cell_11/ReluRelu(simple_rnn_11/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ |
+simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    в
simple_rnn_11/TensorArrayV2_1TensorListReserve4simple_rnn_11/TensorArrayV2_1/element_shape:output:0&simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“T
simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€b
 simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : У
simple_rnn_11/whileWhile)simple_rnn_11/while/loop_counter:output:0/simple_rnn_11/while/maximum_iterations:output:0simple_rnn_11/time:output:0&simple_rnn_11/TensorArrayV2_1:handle:0simple_rnn_11/zeros:output:0&simple_rnn_11/strided_slice_1:output:0Esimple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_11_simple_rnn_cell_11_matmul_readvariableop_resource@simple_rnn_11_simple_rnn_cell_11_biasadd_readvariableop_resourceAsimple_rnn_11_simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_11_while_body_1737717*,
cond$R"
 simple_rnn_11_while_cond_1737716*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations П
>simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    м
0simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_11/while:output:3Gsimple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype0v
#simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€o
%simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ќ
simple_rnn_11/strided_slice_3StridedSlice9simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_11/strided_slice_3/stack:output:0.simple_rnn_11/strided_slice_3/stack_1:output:0.simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_masks
simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ј
simple_rnn_11/transpose_1	Transpose9simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ r
!time_distributed_18/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    £
time_distributed_18/ReshapeReshapesimple_rnn_11/transpose_1:y:0*time_distributed_18/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ѓ
2time_distributed_18/dense_21/MatMul/ReadVariableOpReadVariableOp;time_distributed_18_dense_21_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ѕ
#time_distributed_18/dense_21/MatMulMatMul$time_distributed_18/Reshape:output:0:time_distributed_18/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
3time_distributed_18/dense_21/BiasAdd/ReadVariableOpReadVariableOp<time_distributed_18_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
$time_distributed_18/dense_21/BiasAddBiasAdd-time_distributed_18/dense_21/MatMul:product:0;time_distributed_18/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
#time_distributed_18/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€      ї
time_distributed_18/Reshape_1Reshape-time_distributed_18/dense_21/BiasAdd:output:0,time_distributed_18/Reshape_1/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€t
#time_distributed_18/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    І
time_distributed_18/Reshape_2Reshapesimple_rnn_11/transpose_1:y:0,time_distributed_18/Reshape_2/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ y
IdentityIdentity&time_distributed_18/Reshape_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€≥
NoOpNoOp8^simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOp7^simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOp9^simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOp^simple_rnn_10/while8^simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOp7^simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOp9^simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOp^simple_rnn_11/while6^simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOp5^simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOp7^simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOp^simple_rnn_8/while6^simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOp5^simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOp7^simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOp^simple_rnn_9/while4^time_distributed_18/dense_21/BiasAdd/ReadVariableOp3^time_distributed_18/dense_21/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 2r
7simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOp7simple_rnn_10/simple_rnn_cell_10/BiasAdd/ReadVariableOp2p
6simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOp6simple_rnn_10/simple_rnn_cell_10/MatMul/ReadVariableOp2t
8simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOp8simple_rnn_10/simple_rnn_cell_10/MatMul_1/ReadVariableOp2*
simple_rnn_10/whilesimple_rnn_10/while2r
7simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOp7simple_rnn_11/simple_rnn_cell_11/BiasAdd/ReadVariableOp2p
6simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOp6simple_rnn_11/simple_rnn_cell_11/MatMul/ReadVariableOp2t
8simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOp8simple_rnn_11/simple_rnn_cell_11/MatMul_1/ReadVariableOp2*
simple_rnn_11/whilesimple_rnn_11/while2n
5simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOp5simple_rnn_8/simple_rnn_cell_8/BiasAdd/ReadVariableOp2l
4simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOp4simple_rnn_8/simple_rnn_cell_8/MatMul/ReadVariableOp2p
6simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOp6simple_rnn_8/simple_rnn_cell_8/MatMul_1/ReadVariableOp2(
simple_rnn_8/whilesimple_rnn_8/while2n
5simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOp5simple_rnn_9/simple_rnn_cell_9/BiasAdd/ReadVariableOp2l
4simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOp4simple_rnn_9/simple_rnn_cell_9/MatMul/ReadVariableOp2p
6simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOp6simple_rnn_9/simple_rnn_cell_9/MatMul_1/ReadVariableOp2(
simple_rnn_9/whilesimple_rnn_9/while2j
3time_distributed_18/dense_21/BiasAdd/ReadVariableOp3time_distributed_18/dense_21/BiasAdd/ReadVariableOp2h
2time_distributed_18/dense_21/MatMul/ReadVariableOp2time_distributed_18/dense_21/MatMul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
я
ѓ
while_cond_1739875
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1739875___redundant_placeholder05
1while_while_cond_1739875___redundant_placeholder15
1while_while_cond_1739875___redundant_placeholder25
1while_while_cond_1739875___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
У
к
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_1735503

inputs

states0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€G
ReluReluadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namestates
љ

№
4__inference_simple_rnn_cell_11_layer_call_fn_1740418

inputs
states_0
unknown: 
	unknown_0: 
	unknown_1:  
identity

identity_1ИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_1735675o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/0
љ

№
4__inference_simple_rnn_cell_10_layer_call_fn_1740370

inputs
states_0
unknown:
	unknown_0:
	unknown_1:
identity

identity_1ИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_1735503o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states/0
•=
Љ
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1736114

inputsB
0simple_rnn_cell_8_matmul_readvariableop_resource: ?
1simple_rnn_cell_8_biasadd_readvariableop_resource: D
2simple_rnn_cell_8_matmul_1_readvariableop_resource:  
identityИҐ(simple_rnn_cell_8/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_8/MatMul/ReadVariableOpҐ)simple_rnn_cell_8/MatMul_1/ReadVariableOpҐwhile;
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
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
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
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Я
simple_rnn_cell_8/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ц
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ђ
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ь
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0Щ
simple_rnn_cell_8/MatMul_1MatMulzeros:output:01simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ъ
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ k
simple_rnn_cell_8/ReluRelusimple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource2simple_rnn_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1736048*
condR
while_cond_1736047*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype0h
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
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ ѕ
NoOpNoOp)^simple_rnn_cell_8/BiasAdd/ReadVariableOp(^simple_rnn_cell_8/MatMul/ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2T
(simple_rnn_cell_8/BiasAdd/ReadVariableOp(simple_rnn_cell_8/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_8/MatMul/ReadVariableOp'simple_rnn_cell_8/MatMul/ReadVariableOp2V
)simple_rnn_cell_8/MatMul_1/ReadVariableOp)simple_rnn_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
м,
“
while_body_1739616
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:ИҐ/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_10/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0®
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0≈
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€¶
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ѕ
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0ђ
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ѓ
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€y
while/simple_rnn_cell_10/ReluRelu while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€‘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder+while/simple_rnn_cell_10/Relu:activations:0*
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
: И
while/Identity_4Identity+while/simple_rnn_cell_10/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€в

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
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2b
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
ш9
ц
 simple_rnn_10_while_body_17376138
4simple_rnn_10_while_simple_rnn_10_while_loop_counter>
:simple_rnn_10_while_simple_rnn_10_while_maximum_iterations#
simple_rnn_10_while_placeholder%
!simple_rnn_10_while_placeholder_1%
!simple_rnn_10_while_placeholder_27
3simple_rnn_10_while_simple_rnn_10_strided_slice_1_0s
osimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_10_while_simple_rnn_cell_10_matmul_readvariableop_resource_0:V
Hsimple_rnn_10_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:[
Isimple_rnn_10_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0: 
simple_rnn_10_while_identity"
simple_rnn_10_while_identity_1"
simple_rnn_10_while_identity_2"
simple_rnn_10_while_identity_3"
simple_rnn_10_while_identity_45
1simple_rnn_10_while_simple_rnn_10_strided_slice_1q
msimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_10_while_simple_rnn_cell_10_matmul_readvariableop_resource:T
Fsimple_rnn_10_while_simple_rnn_cell_10_biasadd_readvariableop_resource:Y
Gsimple_rnn_10_while_simple_rnn_cell_10_matmul_1_readvariableop_resource:ИҐ=simple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpҐ<simple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOpҐ>simple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpЦ
Esimple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   м
7simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_10_while_placeholderNsimple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0ƒ
<simple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_10_while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0п
-simple_rnn_10/while/simple_rnn_cell_10/MatMulMatMul>simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€¬
=simple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_10_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0л
.simple_rnn_10/while/simple_rnn_cell_10/BiasAddBiasAdd7simple_rnn_10/while/simple_rnn_cell_10/MatMul:product:0Esimple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€»
>simple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_10_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0÷
/simple_rnn_10/while/simple_rnn_cell_10/MatMul_1MatMul!simple_rnn_10_while_placeholder_2Fsimple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ў
*simple_rnn_10/while/simple_rnn_cell_10/addAddV27simple_rnn_10/while/simple_rnn_cell_10/BiasAdd:output:09simple_rnn_10/while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Х
+simple_rnn_10/while/simple_rnn_cell_10/ReluRelu.simple_rnn_10/while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€М
8simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_10_while_placeholder_1simple_rnn_10_while_placeholder9simple_rnn_10/while/simple_rnn_cell_10/Relu:activations:0*
_output_shapes
: *
element_dtype0:йи“[
simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ж
simple_rnn_10/while/addAddV2simple_rnn_10_while_placeholder"simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Я
simple_rnn_10/while/add_1AddV24simple_rnn_10_while_simple_rnn_10_while_loop_counter$simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: Г
simple_rnn_10/while/IdentityIdentitysimple_rnn_10/while/add_1:z:0^simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Ґ
simple_rnn_10/while/Identity_1Identity:simple_rnn_10_while_simple_rnn_10_while_maximum_iterations^simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Г
simple_rnn_10/while/Identity_2Identitysimple_rnn_10/while/add:z:0^simple_rnn_10/while/NoOp*
T0*
_output_shapes
: ∞
simple_rnn_10/while/Identity_3IdentityHsimple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_10/while/NoOp*
T0*
_output_shapes
: ≤
simple_rnn_10/while/Identity_4Identity9simple_rnn_10/while/simple_rnn_cell_10/Relu:activations:0^simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ъ
simple_rnn_10/while/NoOpNoOp>^simple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp=^simple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOp?^simple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_10_while_identity%simple_rnn_10/while/Identity:output:0"I
simple_rnn_10_while_identity_1'simple_rnn_10/while/Identity_1:output:0"I
simple_rnn_10_while_identity_2'simple_rnn_10/while/Identity_2:output:0"I
simple_rnn_10_while_identity_3'simple_rnn_10/while/Identity_3:output:0"I
simple_rnn_10_while_identity_4'simple_rnn_10/while/Identity_4:output:0"h
1simple_rnn_10_while_simple_rnn_10_strided_slice_13simple_rnn_10_while_simple_rnn_10_strided_slice_1_0"Т
Fsimple_rnn_10_while_simple_rnn_cell_10_biasadd_readvariableop_resourceHsimple_rnn_10_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"Ф
Gsimple_rnn_10_while_simple_rnn_cell_10_matmul_1_readvariableop_resourceIsimple_rnn_10_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"Р
Esimple_rnn_10_while_simple_rnn_cell_10_matmul_readvariableop_resourceGsimple_rnn_10_while_simple_rnn_cell_10_matmul_readvariableop_resource_0"а
msimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorosimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2~
=simple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp=simple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2|
<simple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOp<simple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOp2А
>simple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp>simple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
ƒ
Ч
*__inference_dense_21_layer_call_fn_1740475

inputs
unknown: 
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
E__inference_dense_21_layer_call_and_return_conditional_losses_1735943o
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
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ђ>
Љ
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1739083

inputsB
0simple_rnn_cell_9_matmul_readvariableop_resource: ?
1simple_rnn_cell_9_biasadd_readvariableop_resource:D
2simple_rnn_cell_9_matmul_1_readvariableop_resource:
identityИҐ(simple_rnn_cell_9/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_9/MatMul/ReadVariableOpҐ)simple_rnn_cell_9/MatMul_1/ReadVariableOpҐwhile;
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
value	B :s
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ D
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
valueB"€€€€    а
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
:€€€€€€€€€ *
shrink_axis_maskШ
'simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Я
simple_rnn_cell_9/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ц
(simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ђ
simple_rnn_cell_9/BiasAddBiasAdd"simple_rnn_cell_9/MatMul:product:00simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
)simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Щ
simple_rnn_cell_9/MatMul_1MatMulzeros:output:01simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
simple_rnn_cell_9/addAddV2"simple_rnn_cell_9/BiasAdd:output:0$simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€k
simple_rnn_cell_9/ReluRelusimple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_9_matmul_readvariableop_resource1simple_rnn_cell_9_biasadd_readvariableop_resource2simple_rnn_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1739016*
condR
while_cond_1739015*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
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
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ѕ
NoOpNoOp)^simple_rnn_cell_9/BiasAdd/ReadVariableOp(^simple_rnn_cell_9/MatMul/ReadVariableOp*^simple_rnn_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ : : : 2T
(simple_rnn_cell_9/BiasAdd/ReadVariableOp(simple_rnn_cell_9/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_9/MatMul/ReadVariableOp'simple_rnn_cell_9/MatMul/ReadVariableOp2V
)simple_rnn_cell_9/MatMul_1/ReadVariableOp)simple_rnn_cell_9/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
љ
i
M__inference_repeat_vector_18_layer_call_and_return_conditional_losses_1739206

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
 :€€€€€€€€€€€€€€€€€€Z
stackConst*
_output_shapes
:*
dtype0*!
valueB"         p
TileTileExpandDims:output:0stack:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€b
IdentityIdentityTile:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€€€€€€€€€€:X T
0
_output_shapes
:€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
я
ѓ
while_cond_1736563
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1736563___redundant_placeholder05
1while_while_cond_1736563___redundant_placeholder15
1while_while_cond_1736563___redundant_placeholder25
1while_while_cond_1736563___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
ЩG
ч
-sequential_19_simple_rnn_9_while_body_1734441R
Nsequential_19_simple_rnn_9_while_sequential_19_simple_rnn_9_while_loop_counterX
Tsequential_19_simple_rnn_9_while_sequential_19_simple_rnn_9_while_maximum_iterations0
,sequential_19_simple_rnn_9_while_placeholder2
.sequential_19_simple_rnn_9_while_placeholder_12
.sequential_19_simple_rnn_9_while_placeholder_2Q
Msequential_19_simple_rnn_9_while_sequential_19_simple_rnn_9_strided_slice_1_0О
Йsequential_19_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0e
Ssequential_19_simple_rnn_9_while_simple_rnn_cell_9_matmul_readvariableop_resource_0: b
Tsequential_19_simple_rnn_9_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0:g
Usequential_19_simple_rnn_9_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0:-
)sequential_19_simple_rnn_9_while_identity/
+sequential_19_simple_rnn_9_while_identity_1/
+sequential_19_simple_rnn_9_while_identity_2/
+sequential_19_simple_rnn_9_while_identity_3/
+sequential_19_simple_rnn_9_while_identity_4O
Ksequential_19_simple_rnn_9_while_sequential_19_simple_rnn_9_strided_slice_1М
Зsequential_19_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorc
Qsequential_19_simple_rnn_9_while_simple_rnn_cell_9_matmul_readvariableop_resource: `
Rsequential_19_simple_rnn_9_while_simple_rnn_cell_9_biasadd_readvariableop_resource:e
Ssequential_19_simple_rnn_9_while_simple_rnn_cell_9_matmul_1_readvariableop_resource:ИҐIsequential_19/simple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOpҐHsequential_19/simple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOpҐJsequential_19/simple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp£
Rsequential_19/simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ѓ
Dsequential_19/simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЙsequential_19_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0,sequential_19_simple_rnn_9_while_placeholder[sequential_19/simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0№
Hsequential_19/simple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOpSsequential_19_simple_rnn_9_while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0Ф
9sequential_19/simple_rnn_9/while/simple_rnn_cell_9/MatMulMatMulKsequential_19/simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Psequential_19/simple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Џ
Isequential_19/simple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOpTsequential_19_simple_rnn_9_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0П
:sequential_19/simple_rnn_9/while/simple_rnn_cell_9/BiasAddBiasAddCsequential_19/simple_rnn_9/while/simple_rnn_cell_9/MatMul:product:0Qsequential_19/simple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€а
Jsequential_19/simple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOpUsequential_19_simple_rnn_9_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0ы
;sequential_19/simple_rnn_9/while/simple_rnn_cell_9/MatMul_1MatMul.sequential_19_simple_rnn_9_while_placeholder_2Rsequential_19/simple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€э
6sequential_19/simple_rnn_9/while/simple_rnn_cell_9/addAddV2Csequential_19/simple_rnn_9/while/simple_rnn_cell_9/BiasAdd:output:0Esequential_19/simple_rnn_9/while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€≠
7sequential_19/simple_rnn_9/while/simple_rnn_cell_9/ReluRelu:sequential_19/simple_rnn_9/while/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€Н
Ksequential_19/simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : з
Esequential_19/simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_19_simple_rnn_9_while_placeholder_1Tsequential_19/simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:0Esequential_19/simple_rnn_9/while/simple_rnn_cell_9/Relu:activations:0*
_output_shapes
: *
element_dtype0:йи“h
&sequential_19/simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :≠
$sequential_19/simple_rnn_9/while/addAddV2,sequential_19_simple_rnn_9_while_placeholder/sequential_19/simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: j
(sequential_19/simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :”
&sequential_19/simple_rnn_9/while/add_1AddV2Nsequential_19_simple_rnn_9_while_sequential_19_simple_rnn_9_while_loop_counter1sequential_19/simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: ™
)sequential_19/simple_rnn_9/while/IdentityIdentity*sequential_19/simple_rnn_9/while/add_1:z:0&^sequential_19/simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ÷
+sequential_19/simple_rnn_9/while/Identity_1IdentityTsequential_19_simple_rnn_9_while_sequential_19_simple_rnn_9_while_maximum_iterations&^sequential_19/simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ™
+sequential_19/simple_rnn_9/while/Identity_2Identity(sequential_19/simple_rnn_9/while/add:z:0&^sequential_19/simple_rnn_9/while/NoOp*
T0*
_output_shapes
: „
+sequential_19/simple_rnn_9/while/Identity_3IdentityUsequential_19/simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^sequential_19/simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Ў
+sequential_19/simple_rnn_9/while/Identity_4IdentityEsequential_19/simple_rnn_9/while/simple_rnn_cell_9/Relu:activations:0&^sequential_19/simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ћ
%sequential_19/simple_rnn_9/while/NoOpNoOpJ^sequential_19/simple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOpI^sequential_19/simple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOpK^sequential_19/simple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "_
)sequential_19_simple_rnn_9_while_identity2sequential_19/simple_rnn_9/while/Identity:output:0"c
+sequential_19_simple_rnn_9_while_identity_14sequential_19/simple_rnn_9/while/Identity_1:output:0"c
+sequential_19_simple_rnn_9_while_identity_24sequential_19/simple_rnn_9/while/Identity_2:output:0"c
+sequential_19_simple_rnn_9_while_identity_34sequential_19/simple_rnn_9/while/Identity_3:output:0"c
+sequential_19_simple_rnn_9_while_identity_44sequential_19/simple_rnn_9/while/Identity_4:output:0"Ь
Ksequential_19_simple_rnn_9_while_sequential_19_simple_rnn_9_strided_slice_1Msequential_19_simple_rnn_9_while_sequential_19_simple_rnn_9_strided_slice_1_0"™
Rsequential_19_simple_rnn_9_while_simple_rnn_cell_9_biasadd_readvariableop_resourceTsequential_19_simple_rnn_9_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"ђ
Ssequential_19_simple_rnn_9_while_simple_rnn_cell_9_matmul_1_readvariableop_resourceUsequential_19_simple_rnn_9_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"®
Qsequential_19_simple_rnn_9_while_simple_rnn_cell_9_matmul_readvariableop_resourceSsequential_19_simple_rnn_9_while_simple_rnn_cell_9_matmul_readvariableop_resource_0"Ц
Зsequential_19_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorЙsequential_19_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2Ц
Isequential_19/simple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOpIsequential_19/simple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2Ф
Hsequential_19/simple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOpHsequential_19/simple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOp2Ш
Jsequential_19/simple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpJsequential_19/simple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
Є
÷
P__inference_time_distributed_18_layer_call_and_return_conditional_losses_1735954

inputs"
dense_21_1735944: 
dense_21_1735946:
identityИҐ dense_21/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
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
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ э
 dense_21/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_21_1735944dense_21_1735946*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_1735943\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:Ш
	Reshape_1Reshape)dense_21/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€i
NoOpNoOp!^dense_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€ : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
ї

џ
3__inference_simple_rnn_cell_9_layer_call_fn_1740308

inputs
states_0
unknown: 
	unknown_0:
	unknown_1:
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
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_1735194o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€ :€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states/0
д
л
/__inference_sequential_19_layer_call_fn_1737357

inputs
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8: 
	unknown_9: 

unknown_10:  

unknown_11: 

unknown_12:
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_19_layer_call_and_return_conditional_losses_1737106s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ш9
ц
 simple_rnn_10_while_body_17380518
4simple_rnn_10_while_simple_rnn_10_while_loop_counter>
:simple_rnn_10_while_simple_rnn_10_while_maximum_iterations#
simple_rnn_10_while_placeholder%
!simple_rnn_10_while_placeholder_1%
!simple_rnn_10_while_placeholder_27
3simple_rnn_10_while_simple_rnn_10_strided_slice_1_0s
osimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_10_while_simple_rnn_cell_10_matmul_readvariableop_resource_0:V
Hsimple_rnn_10_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:[
Isimple_rnn_10_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0: 
simple_rnn_10_while_identity"
simple_rnn_10_while_identity_1"
simple_rnn_10_while_identity_2"
simple_rnn_10_while_identity_3"
simple_rnn_10_while_identity_45
1simple_rnn_10_while_simple_rnn_10_strided_slice_1q
msimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_10_while_simple_rnn_cell_10_matmul_readvariableop_resource:T
Fsimple_rnn_10_while_simple_rnn_cell_10_biasadd_readvariableop_resource:Y
Gsimple_rnn_10_while_simple_rnn_cell_10_matmul_1_readvariableop_resource:ИҐ=simple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpҐ<simple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOpҐ>simple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpЦ
Esimple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   м
7simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_10_while_placeholderNsimple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0ƒ
<simple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_10_while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0п
-simple_rnn_10/while/simple_rnn_cell_10/MatMulMatMul>simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€¬
=simple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_10_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0л
.simple_rnn_10/while/simple_rnn_cell_10/BiasAddBiasAdd7simple_rnn_10/while/simple_rnn_cell_10/MatMul:product:0Esimple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€»
>simple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_10_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0÷
/simple_rnn_10/while/simple_rnn_cell_10/MatMul_1MatMul!simple_rnn_10_while_placeholder_2Fsimple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ў
*simple_rnn_10/while/simple_rnn_cell_10/addAddV27simple_rnn_10/while/simple_rnn_cell_10/BiasAdd:output:09simple_rnn_10/while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Х
+simple_rnn_10/while/simple_rnn_cell_10/ReluRelu.simple_rnn_10/while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€М
8simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_10_while_placeholder_1simple_rnn_10_while_placeholder9simple_rnn_10/while/simple_rnn_cell_10/Relu:activations:0*
_output_shapes
: *
element_dtype0:йи“[
simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ж
simple_rnn_10/while/addAddV2simple_rnn_10_while_placeholder"simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Я
simple_rnn_10/while/add_1AddV24simple_rnn_10_while_simple_rnn_10_while_loop_counter$simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: Г
simple_rnn_10/while/IdentityIdentitysimple_rnn_10/while/add_1:z:0^simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Ґ
simple_rnn_10/while/Identity_1Identity:simple_rnn_10_while_simple_rnn_10_while_maximum_iterations^simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Г
simple_rnn_10/while/Identity_2Identitysimple_rnn_10/while/add:z:0^simple_rnn_10/while/NoOp*
T0*
_output_shapes
: ∞
simple_rnn_10/while/Identity_3IdentityHsimple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_10/while/NoOp*
T0*
_output_shapes
: ≤
simple_rnn_10/while/Identity_4Identity9simple_rnn_10/while/simple_rnn_cell_10/Relu:activations:0^simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ъ
simple_rnn_10/while/NoOpNoOp>^simple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp=^simple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOp?^simple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_10_while_identity%simple_rnn_10/while/Identity:output:0"I
simple_rnn_10_while_identity_1'simple_rnn_10/while/Identity_1:output:0"I
simple_rnn_10_while_identity_2'simple_rnn_10/while/Identity_2:output:0"I
simple_rnn_10_while_identity_3'simple_rnn_10/while/Identity_3:output:0"I
simple_rnn_10_while_identity_4'simple_rnn_10/while/Identity_4:output:0"h
1simple_rnn_10_while_simple_rnn_10_strided_slice_13simple_rnn_10_while_simple_rnn_10_strided_slice_1_0"Т
Fsimple_rnn_10_while_simple_rnn_cell_10_biasadd_readvariableop_resourceHsimple_rnn_10_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"Ф
Gsimple_rnn_10_while_simple_rnn_cell_10_matmul_1_readvariableop_resourceIsimple_rnn_10_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"Р
Esimple_rnn_10_while_simple_rnn_cell_10_matmul_readvariableop_resourceGsimple_rnn_10_while_simple_rnn_cell_10_matmul_readvariableop_resource_0"а
msimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorosimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2~
=simple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp=simple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2|
<simple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOp<simple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOp2А
>simple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp>simple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
Ш
л
N__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_1740280

inputs
states_00
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 2
 matmul_1_readvariableop_resource:  
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ G
ReluReluadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/0
Е5
Я
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1735311

inputs+
simple_rnn_cell_9_1735234: '
simple_rnn_cell_9_1735236:+
simple_rnn_cell_9_1735238:
identityИҐ)simple_rnn_cell_9/StatefulPartitionedCallҐwhile;
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
value	B :s
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ D
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
valueB"€€€€    а
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
:€€€€€€€€€ *
shrink_axis_maskл
)simple_rnn_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_9_1735234simple_rnn_cell_9_1735236simple_rnn_cell_9_1735238*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_1735194n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_9_1735234simple_rnn_cell_9_1735236simple_rnn_cell_9_1735238*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1735247*
condR
while_cond_1735246*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
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
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€z
NoOpNoOp*^simple_rnn_cell_9/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 2V
)simple_rnn_cell_9/StatefulPartitionedCall)simple_rnn_cell_9/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
э9
ѕ
simple_rnn_9_while_body_17379426
2simple_rnn_9_while_simple_rnn_9_while_loop_counter<
8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations"
simple_rnn_9_while_placeholder$
 simple_rnn_9_while_placeholder_1$
 simple_rnn_9_while_placeholder_25
1simple_rnn_9_while_simple_rnn_9_strided_slice_1_0q
msimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_9_while_simple_rnn_cell_9_matmul_readvariableop_resource_0: T
Fsimple_rnn_9_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0:Y
Gsimple_rnn_9_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0:
simple_rnn_9_while_identity!
simple_rnn_9_while_identity_1!
simple_rnn_9_while_identity_2!
simple_rnn_9_while_identity_3!
simple_rnn_9_while_identity_43
/simple_rnn_9_while_simple_rnn_9_strided_slice_1o
ksimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_9_while_simple_rnn_cell_9_matmul_readvariableop_resource: R
Dsimple_rnn_9_while_simple_rnn_cell_9_biasadd_readvariableop_resource:W
Esimple_rnn_9_while_simple_rnn_cell_9_matmul_1_readvariableop_resource:ИҐ;simple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOpҐ:simple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOpҐ<simple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpХ
Dsimple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    з
6simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_9_while_placeholderMsimple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0ј
:simple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_9_while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0к
+simple_rnn_9/while/simple_rnn_cell_9/MatMulMatMul=simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Њ
;simple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_9_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0е
,simple_rnn_9/while/simple_rnn_cell_9/BiasAddBiasAdd5simple_rnn_9/while/simple_rnn_cell_9/MatMul:product:0Csimple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ƒ
<simple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_9_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0—
-simple_rnn_9/while/simple_rnn_cell_9/MatMul_1MatMul simple_rnn_9_while_placeholder_2Dsimple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€”
(simple_rnn_9/while/simple_rnn_cell_9/addAddV25simple_rnn_9/while/simple_rnn_cell_9/BiasAdd:output:07simple_rnn_9/while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€С
)simple_rnn_9/while/simple_rnn_cell_9/ReluRelu,simple_rnn_9/while/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€
=simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ѓ
7simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_9_while_placeholder_1Fsimple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:07simple_rnn_9/while/simple_rnn_cell_9/Relu:activations:0*
_output_shapes
: *
element_dtype0:йи“Z
simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Г
simple_rnn_9/while/addAddV2simple_rnn_9_while_placeholder!simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ы
simple_rnn_9/while/add_1AddV22simple_rnn_9_while_simple_rnn_9_while_loop_counter#simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: А
simple_rnn_9/while/IdentityIdentitysimple_rnn_9/while/add_1:z:0^simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Ю
simple_rnn_9/while/Identity_1Identity8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations^simple_rnn_9/while/NoOp*
T0*
_output_shapes
: А
simple_rnn_9/while/Identity_2Identitysimple_rnn_9/while/add:z:0^simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ≠
simple_rnn_9/while/Identity_3IdentityGsimple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Ѓ
simple_rnn_9/while/Identity_4Identity7simple_rnn_9/while/simple_rnn_cell_9/Relu:activations:0^simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€У
simple_rnn_9/while/NoOpNoOp<^simple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp;^simple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOp=^simple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_9_while_identity$simple_rnn_9/while/Identity:output:0"G
simple_rnn_9_while_identity_1&simple_rnn_9/while/Identity_1:output:0"G
simple_rnn_9_while_identity_2&simple_rnn_9/while/Identity_2:output:0"G
simple_rnn_9_while_identity_3&simple_rnn_9/while/Identity_3:output:0"G
simple_rnn_9_while_identity_4&simple_rnn_9/while/Identity_4:output:0"d
/simple_rnn_9_while_simple_rnn_9_strided_slice_11simple_rnn_9_while_simple_rnn_9_strided_slice_1_0"О
Dsimple_rnn_9_while_simple_rnn_cell_9_biasadd_readvariableop_resourceFsimple_rnn_9_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"Р
Esimple_rnn_9_while_simple_rnn_cell_9_matmul_1_readvariableop_resourceGsimple_rnn_9_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"М
Csimple_rnn_9_while_simple_rnn_cell_9_matmul_readvariableop_resourceEsimple_rnn_9_while_simple_rnn_cell_9_matmul_readvariableop_resource_0"№
ksimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensormsimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2z
;simple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp;simple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2x
:simple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOp:simple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOp2|
<simple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp<simple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
Ю!
я
while_body_1735396
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_10_1735418_0:0
"while_simple_rnn_cell_10_1735420_0:4
"while_simple_rnn_cell_10_1735422_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_10_1735418:.
 while_simple_rnn_cell_10_1735420:2
 while_simple_rnn_cell_10_1735422:ИҐ0while/simple_rnn_cell_10/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Ђ
0while/simple_rnn_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_10_1735418_0"while_simple_rnn_cell_10_1735420_0"while_simple_rnn_cell_10_1735422_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_1735383в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_10/StatefulPartitionedCall:output:0*
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
: Ц
while/Identity_4Identity9while/simple_rnn_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€

while/NoOpNoOp1^while/simple_rnn_cell_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_10_1735418"while_simple_rnn_cell_10_1735418_0"F
 while_simple_rnn_cell_10_1735420"while_simple_rnn_cell_10_1735420_0"F
 while_simple_rnn_cell_10_1735422"while_simple_rnn_cell_10_1735422_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2d
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
З!
Ў
while_body_1734952
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
!while_simple_rnn_cell_8_1734974_0: /
!while_simple_rnn_cell_8_1734976_0: 3
!while_simple_rnn_cell_8_1734978_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
while_simple_rnn_cell_8_1734974: -
while_simple_rnn_cell_8_1734976: 1
while_simple_rnn_cell_8_1734978:  ИҐ/while/simple_rnn_cell_8/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0¶
/while/simple_rnn_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_8_1734974_0!while_simple_rnn_cell_8_1734976_0!while_simple_rnn_cell_8_1734978_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_1734900б
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_8/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity8while/simple_rnn_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ~

while/NoOpNoOp0^while/simple_rnn_cell_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_8_1734974!while_simple_rnn_cell_8_1734974_0"D
while_simple_rnn_cell_8_1734976!while_simple_rnn_cell_8_1734976_0"D
while_simple_rnn_cell_8_1734978!while_simple_rnn_cell_8_1734978_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2b
/while/simple_rnn_cell_8/StatefulPartitionedCall/while/simple_rnn_cell_8/StatefulPartitionedCall: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
≤
Ї
.__inference_simple_rnn_8_layer_call_fn_1738255
inputs_0
unknown: 
	unknown_0: 
	unknown_1:  
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1735015|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ `
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
љ
i
M__inference_repeat_vector_18_layer_call_and_return_conditional_losses_1735332

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
 :€€€€€€€€€€€€€€€€€€Z
stackConst*
_output_shapes
:*
dtype0*!
valueB"         p
TileTileExpandDims:output:0stack:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€b
IdentityIdentityTile:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€€€€€€€€€€:X T
0
_output_shapes
:€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
»	
ц
E__inference_dense_21_layer_call_and_return_conditional_losses_1735943

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
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
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ю!
я
while_body_1735688
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_11_1735710_0: 0
"while_simple_rnn_cell_11_1735712_0: 4
"while_simple_rnn_cell_11_1735714_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_11_1735710: .
 while_simple_rnn_cell_11_1735712: 2
 while_simple_rnn_cell_11_1735714:  ИҐ0while/simple_rnn_cell_11/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Ђ
0while/simple_rnn_cell_11/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_11_1735710_0"while_simple_rnn_cell_11_1735712_0"while_simple_rnn_cell_11_1735714_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_1735675в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_11/StatefulPartitionedCall:output:0*
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
: Ц
while/Identity_4Identity9while/simple_rnn_cell_11/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 

while/NoOpNoOp1^while/simple_rnn_cell_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_11_1735710"while_simple_rnn_cell_11_1735710_0"F
 while_simple_rnn_cell_11_1735712"while_simple_rnn_cell_11_1735712_0"F
 while_simple_rnn_cell_11_1735714"while_simple_rnn_cell_11_1735714_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2d
0while/simple_rnn_cell_11/StatefulPartitionedCall0while/simple_rnn_cell_11/StatefulPartitionedCall: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Ъ4
Я
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1735015

inputs+
simple_rnn_cell_8_1734940: '
simple_rnn_cell_8_1734942: +
simple_rnn_cell_8_1734944:  
identityИҐ)simple_rnn_cell_8/StatefulPartitionedCallҐwhile;
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
:€€€€€€€€€ c
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
)simple_rnn_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_8_1734940simple_rnn_cell_8_1734942simple_rnn_cell_8_1734944*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_1734900n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_8_1734940simple_rnn_cell_8_1734942simple_rnn_cell_8_1734944*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1734952*
condR
while_cond_1734951*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
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
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ z
NoOpNoOp*^simple_rnn_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2V
)simple_rnn_cell_8/StatefulPartitionedCall)simple_rnn_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
З!
Ў
while_body_1734793
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
!while_simple_rnn_cell_8_1734815_0: /
!while_simple_rnn_cell_8_1734817_0: 3
!while_simple_rnn_cell_8_1734819_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
while_simple_rnn_cell_8_1734815: -
while_simple_rnn_cell_8_1734817: 1
while_simple_rnn_cell_8_1734819:  ИҐ/while/simple_rnn_cell_8/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0¶
/while/simple_rnn_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_8_1734815_0!while_simple_rnn_cell_8_1734817_0!while_simple_rnn_cell_8_1734819_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_1734780б
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_8/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity8while/simple_rnn_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ~

while/NoOpNoOp0^while/simple_rnn_cell_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_8_1734815!while_simple_rnn_cell_8_1734815_0"D
while_simple_rnn_cell_8_1734817!while_simple_rnn_cell_8_1734817_0"D
while_simple_rnn_cell_8_1734819!while_simple_rnn_cell_8_1734819_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2b
/while/simple_rnn_cell_8/StatefulPartitionedCall/while/simple_rnn_cell_8/StatefulPartitionedCall: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
я
ѓ
while_cond_1739615
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1739615___redundant_placeholder05
1while_while_cond_1739615___redundant_placeholder15
1while_while_cond_1739615___redundant_placeholder25
1while_while_cond_1739615___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
м,
“
while_body_1739292
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:ИҐ/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_10/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0®
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0≈
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€¶
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ѕ
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0ђ
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ѓ
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€y
while/simple_rnn_cell_10/ReluRelu while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€‘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder+while/simple_rnn_cell_10/Relu:activations:0*
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
: И
while/Identity_4Identity+while/simple_rnn_cell_10/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€в

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
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2b
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
Ю!
я
while_body_1735847
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_11_1735869_0: 0
"while_simple_rnn_cell_11_1735871_0: 4
"while_simple_rnn_cell_11_1735873_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_11_1735869: .
 while_simple_rnn_cell_11_1735871: 2
 while_simple_rnn_cell_11_1735873:  ИҐ0while/simple_rnn_cell_11/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Ђ
0while/simple_rnn_cell_11/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_11_1735869_0"while_simple_rnn_cell_11_1735871_0"while_simple_rnn_cell_11_1735873_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_1735795в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_11/StatefulPartitionedCall:output:0*
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
: Ц
while/Identity_4Identity9while/simple_rnn_cell_11/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 

while/NoOpNoOp1^while/simple_rnn_cell_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_11_1735869"while_simple_rnn_cell_11_1735869_0"F
 while_simple_rnn_cell_11_1735871"while_simple_rnn_cell_11_1735871_0"F
 while_simple_rnn_cell_11_1735873"while_simple_rnn_cell_11_1735873_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2d
0while/simple_rnn_cell_11/StatefulPartitionedCall0while/simple_rnn_cell_11/StatefulPartitionedCall: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
≤
Ї
.__inference_simple_rnn_8_layer_call_fn_1738244
inputs_0
unknown: 
	unknown_0: 
	unknown_1:  
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1734856|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ `
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
‘8
ѕ
simple_rnn_8_while_body_17373996
2simple_rnn_8_while_simple_rnn_8_while_loop_counter<
8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations"
simple_rnn_8_while_placeholder$
 simple_rnn_8_while_placeholder_1$
 simple_rnn_8_while_placeholder_25
1simple_rnn_8_while_simple_rnn_8_strided_slice_1_0q
msimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_8_while_simple_rnn_cell_8_matmul_readvariableop_resource_0: T
Fsimple_rnn_8_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0: Y
Gsimple_rnn_8_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0:  
simple_rnn_8_while_identity!
simple_rnn_8_while_identity_1!
simple_rnn_8_while_identity_2!
simple_rnn_8_while_identity_3!
simple_rnn_8_while_identity_43
/simple_rnn_8_while_simple_rnn_8_strided_slice_1o
ksimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_8_while_simple_rnn_cell_8_matmul_readvariableop_resource: R
Dsimple_rnn_8_while_simple_rnn_cell_8_biasadd_readvariableop_resource: W
Esimple_rnn_8_while_simple_rnn_cell_8_matmul_1_readvariableop_resource:  ИҐ;simple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpҐ:simple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOpҐ<simple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpХ
Dsimple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   з
6simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_8_while_placeholderMsimple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0ј
:simple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_8_while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0к
+simple_rnn_8/while/simple_rnn_cell_8/MatMulMatMul=simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Њ
;simple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_8_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0е
,simple_rnn_8/while/simple_rnn_cell_8/BiasAddBiasAdd5simple_rnn_8/while/simple_rnn_cell_8/MatMul:product:0Csimple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ƒ
<simple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_8_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0—
-simple_rnn_8/while/simple_rnn_cell_8/MatMul_1MatMul simple_rnn_8_while_placeholder_2Dsimple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ”
(simple_rnn_8/while/simple_rnn_cell_8/addAddV25simple_rnn_8/while/simple_rnn_cell_8/BiasAdd:output:07simple_rnn_8/while/simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ С
)simple_rnn_8/while/simple_rnn_cell_8/ReluRelu,simple_rnn_8/while/simple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ З
7simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_8_while_placeholder_1simple_rnn_8_while_placeholder7simple_rnn_8/while/simple_rnn_cell_8/Relu:activations:0*
_output_shapes
: *
element_dtype0:йи“Z
simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Г
simple_rnn_8/while/addAddV2simple_rnn_8_while_placeholder!simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ы
simple_rnn_8/while/add_1AddV22simple_rnn_8_while_simple_rnn_8_while_loop_counter#simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: А
simple_rnn_8/while/IdentityIdentitysimple_rnn_8/while/add_1:z:0^simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Ю
simple_rnn_8/while/Identity_1Identity8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations^simple_rnn_8/while/NoOp*
T0*
_output_shapes
: А
simple_rnn_8/while/Identity_2Identitysimple_rnn_8/while/add:z:0^simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ≠
simple_rnn_8/while/Identity_3IdentityGsimple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Ѓ
simple_rnn_8/while/Identity_4Identity7simple_rnn_8/while/simple_rnn_cell_8/Relu:activations:0^simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ У
simple_rnn_8/while/NoOpNoOp<^simple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp;^simple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOp=^simple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_8_while_identity$simple_rnn_8/while/Identity:output:0"G
simple_rnn_8_while_identity_1&simple_rnn_8/while/Identity_1:output:0"G
simple_rnn_8_while_identity_2&simple_rnn_8/while/Identity_2:output:0"G
simple_rnn_8_while_identity_3&simple_rnn_8/while/Identity_3:output:0"G
simple_rnn_8_while_identity_4&simple_rnn_8/while/Identity_4:output:0"d
/simple_rnn_8_while_simple_rnn_8_strided_slice_11simple_rnn_8_while_simple_rnn_8_strided_slice_1_0"О
Dsimple_rnn_8_while_simple_rnn_cell_8_biasadd_readvariableop_resourceFsimple_rnn_8_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"Р
Esimple_rnn_8_while_simple_rnn_cell_8_matmul_1_readvariableop_resourceGsimple_rnn_8_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"М
Csimple_rnn_8_while_simple_rnn_cell_8_matmul_readvariableop_resourceEsimple_rnn_8_while_simple_rnn_cell_8_matmul_readvariableop_resource_0"№
ksimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensormsimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2z
;simple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp;simple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2x
:simple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOp:simple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOp2|
<simple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp<simple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
ш9
ц
 simple_rnn_11_while_body_17377178
4simple_rnn_11_while_simple_rnn_11_while_loop_counter>
:simple_rnn_11_while_simple_rnn_11_while_maximum_iterations#
simple_rnn_11_while_placeholder%
!simple_rnn_11_while_placeholder_1%
!simple_rnn_11_while_placeholder_27
3simple_rnn_11_while_simple_rnn_11_strided_slice_1_0s
osimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_11_while_simple_rnn_cell_11_matmul_readvariableop_resource_0: V
Hsimple_rnn_11_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0: [
Isimple_rnn_11_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:   
simple_rnn_11_while_identity"
simple_rnn_11_while_identity_1"
simple_rnn_11_while_identity_2"
simple_rnn_11_while_identity_3"
simple_rnn_11_while_identity_45
1simple_rnn_11_while_simple_rnn_11_strided_slice_1q
msimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_11_while_simple_rnn_cell_11_matmul_readvariableop_resource: T
Fsimple_rnn_11_while_simple_rnn_cell_11_biasadd_readvariableop_resource: Y
Gsimple_rnn_11_while_simple_rnn_cell_11_matmul_1_readvariableop_resource:  ИҐ=simple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpҐ<simple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOpҐ>simple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpЦ
Esimple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   м
7simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_11_while_placeholderNsimple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0ƒ
<simple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_11_while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0п
-simple_rnn_11/while/simple_rnn_cell_11/MatMulMatMul>simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ¬
=simple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_11_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0л
.simple_rnn_11/while/simple_rnn_cell_11/BiasAddBiasAdd7simple_rnn_11/while/simple_rnn_cell_11/MatMul:product:0Esimple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ »
>simple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_11_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0÷
/simple_rnn_11/while/simple_rnn_cell_11/MatMul_1MatMul!simple_rnn_11_while_placeholder_2Fsimple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ў
*simple_rnn_11/while/simple_rnn_cell_11/addAddV27simple_rnn_11/while/simple_rnn_cell_11/BiasAdd:output:09simple_rnn_11/while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
+simple_rnn_11/while/simple_rnn_cell_11/ReluRelu.simple_rnn_11/while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ М
8simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_11_while_placeholder_1simple_rnn_11_while_placeholder9simple_rnn_11/while/simple_rnn_cell_11/Relu:activations:0*
_output_shapes
: *
element_dtype0:йи“[
simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ж
simple_rnn_11/while/addAddV2simple_rnn_11_while_placeholder"simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Я
simple_rnn_11/while/add_1AddV24simple_rnn_11_while_simple_rnn_11_while_loop_counter$simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: Г
simple_rnn_11/while/IdentityIdentitysimple_rnn_11/while/add_1:z:0^simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Ґ
simple_rnn_11/while/Identity_1Identity:simple_rnn_11_while_simple_rnn_11_while_maximum_iterations^simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Г
simple_rnn_11/while/Identity_2Identitysimple_rnn_11/while/add:z:0^simple_rnn_11/while/NoOp*
T0*
_output_shapes
: ∞
simple_rnn_11/while/Identity_3IdentityHsimple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_11/while/NoOp*
T0*
_output_shapes
: ≤
simple_rnn_11/while/Identity_4Identity9simple_rnn_11/while/simple_rnn_cell_11/Relu:activations:0^simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Ъ
simple_rnn_11/while/NoOpNoOp>^simple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp=^simple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOp?^simple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_11_while_identity%simple_rnn_11/while/Identity:output:0"I
simple_rnn_11_while_identity_1'simple_rnn_11/while/Identity_1:output:0"I
simple_rnn_11_while_identity_2'simple_rnn_11/while/Identity_2:output:0"I
simple_rnn_11_while_identity_3'simple_rnn_11/while/Identity_3:output:0"I
simple_rnn_11_while_identity_4'simple_rnn_11/while/Identity_4:output:0"h
1simple_rnn_11_while_simple_rnn_11_strided_slice_13simple_rnn_11_while_simple_rnn_11_strided_slice_1_0"Т
Fsimple_rnn_11_while_simple_rnn_cell_11_biasadd_readvariableop_resourceHsimple_rnn_11_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"Ф
Gsimple_rnn_11_while_simple_rnn_cell_11_matmul_1_readvariableop_resourceIsimple_rnn_11_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"Р
Esimple_rnn_11_while_simple_rnn_cell_11_matmul_readvariableop_resourceGsimple_rnn_11_while_simple_rnn_cell_11_matmul_readvariableop_resource_0"а
msimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorosimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2~
=simple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp=simple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2|
<simple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOp<simple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOp2А
>simple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp>simple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
і
ї
/__inference_simple_rnn_10_layer_call_fn_1739217
inputs_0
unknown:
	unknown_0:
	unknown_1:
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1735459|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
я
ѓ
while_cond_1739767
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1739767___redundant_placeholder05
1while_while_cond_1739767___redundant_placeholder15
1while_while_cond_1739767___redundant_placeholder25
1while_while_cond_1739767___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
’
√
.sequential_19_simple_rnn_10_while_cond_1734549T
Psequential_19_simple_rnn_10_while_sequential_19_simple_rnn_10_while_loop_counterZ
Vsequential_19_simple_rnn_10_while_sequential_19_simple_rnn_10_while_maximum_iterations1
-sequential_19_simple_rnn_10_while_placeholder3
/sequential_19_simple_rnn_10_while_placeholder_13
/sequential_19_simple_rnn_10_while_placeholder_2V
Rsequential_19_simple_rnn_10_while_less_sequential_19_simple_rnn_10_strided_slice_1m
isequential_19_simple_rnn_10_while_sequential_19_simple_rnn_10_while_cond_1734549___redundant_placeholder0m
isequential_19_simple_rnn_10_while_sequential_19_simple_rnn_10_while_cond_1734549___redundant_placeholder1m
isequential_19_simple_rnn_10_while_sequential_19_simple_rnn_10_while_cond_1734549___redundant_placeholder2m
isequential_19_simple_rnn_10_while_sequential_19_simple_rnn_10_while_cond_1734549___redundant_placeholder3.
*sequential_19_simple_rnn_10_while_identity
“
&sequential_19/simple_rnn_10/while/LessLess-sequential_19_simple_rnn_10_while_placeholderRsequential_19_simple_rnn_10_while_less_sequential_19_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: Г
*sequential_19/simple_rnn_10/while/IdentityIdentity*sequential_19/simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "a
*sequential_19_simple_rnn_10_while_identity3sequential_19/simple_rnn_10/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
К
є
/__inference_simple_rnn_11_layer_call_fn_1739715

inputs
unknown: 
	unknown_0: 
	unknown_1:  
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1736462s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
і
ї
/__inference_simple_rnn_11_layer_call_fn_1739704
inputs_0
unknown: 
	unknown_0: 
	unknown_1:  
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1735910|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
У
к
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_1735383

inputs

states0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€G
ReluReluadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namestates
Т
й
N__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_1734900

inputs

states0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 2
 matmul_1_readvariableop_resource:  
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ G
ReluReluadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
њ

¶
simple_rnn_9_while_cond_17375036
2simple_rnn_9_while_simple_rnn_9_while_loop_counter<
8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations"
simple_rnn_9_while_placeholder$
 simple_rnn_9_while_placeholder_1$
 simple_rnn_9_while_placeholder_28
4simple_rnn_9_while_less_simple_rnn_9_strided_slice_1O
Ksimple_rnn_9_while_simple_rnn_9_while_cond_1737503___redundant_placeholder0O
Ksimple_rnn_9_while_simple_rnn_9_while_cond_1737503___redundant_placeholder1O
Ksimple_rnn_9_while_simple_rnn_9_while_cond_1737503___redundant_placeholder2O
Ksimple_rnn_9_while_simple_rnn_9_while_cond_1737503___redundant_placeholder3
simple_rnn_9_while_identity
Ц
simple_rnn_9/while/LessLesssimple_rnn_9_while_placeholder4simple_rnn_9_while_less_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_9/while/IdentityIdentitysimple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_9_while_identity$simple_rnn_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
 =
√
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1739682

inputsC
1simple_rnn_cell_10_matmul_readvariableop_resource:@
2simple_rnn_cell_10_biasadd_readvariableop_resource:E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:
identityИҐ)simple_rnn_cell_10/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_10/MatMul/ReadVariableOpҐ*simple_rnn_cell_10/MatMul_1/ReadVariableOpҐwhile;
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
value	B :s
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
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
valueB"€€€€   а
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
:€€€€€€€€€*
shrink_axis_maskЪ
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype0°
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Ы
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€m
simple_rnn_cell_10/ReluRelusimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1739616*
condR
while_cond_1739615*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0h
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
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€“
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Т
й
N__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_1735194

inputs

states0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€G
ReluReluadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€ :€€€€€€€€€: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namestates
Т
й
N__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_1734780

inputs

states0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 2
 matmul_1_readvariableop_resource:  
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ G
ReluReluadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
я
ѓ
while_cond_1738534
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1738534___redundant_placeholder05
1while_while_cond_1738534___redundant_placeholder15
1while_while_cond_1738534___redundant_placeholder25
1while_while_cond_1738534___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
я
ѓ
while_cond_1736955
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1736955___redundant_placeholder05
1while_while_cond_1736955___redundant_placeholder15
1while_while_cond_1736955___redundant_placeholder25
1while_while_cond_1736955___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
њ

¶
simple_rnn_9_while_cond_17379416
2simple_rnn_9_while_simple_rnn_9_while_loop_counter<
8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations"
simple_rnn_9_while_placeholder$
 simple_rnn_9_while_placeholder_1$
 simple_rnn_9_while_placeholder_28
4simple_rnn_9_while_less_simple_rnn_9_strided_slice_1O
Ksimple_rnn_9_while_simple_rnn_9_while_cond_1737941___redundant_placeholder0O
Ksimple_rnn_9_while_simple_rnn_9_while_cond_1737941___redundant_placeholder1O
Ksimple_rnn_9_while_simple_rnn_9_while_cond_1737941___redundant_placeholder2O
Ksimple_rnn_9_while_simple_rnn_9_while_cond_1737941___redundant_placeholder3
simple_rnn_9_while_identity
Ц
simple_rnn_9/while/LessLesssimple_rnn_9_while_placeholder4simple_rnn_9_while_less_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_9/while/IdentityIdentitysimple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_9_while_identity$simple_rnn_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
я
ѓ
while_cond_1736824
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1736824___redundant_placeholder05
1while_while_cond_1736824___redundant_placeholder15
1while_while_cond_1736824___redundant_placeholder25
1while_while_cond_1736824___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Щ
м
O__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_1740449

inputs
states_00
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 2
 matmul_1_readvariableop_resource:  
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ G
ReluReluadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/0
•=
Љ
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1738601

inputsB
0simple_rnn_cell_8_matmul_readvariableop_resource: ?
1simple_rnn_cell_8_biasadd_readvariableop_resource: D
2simple_rnn_cell_8_matmul_1_readvariableop_resource:  
identityИҐ(simple_rnn_cell_8/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_8/MatMul/ReadVariableOpҐ)simple_rnn_cell_8/MatMul_1/ReadVariableOpҐwhile;
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
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
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
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Я
simple_rnn_cell_8/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ц
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ђ
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ь
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0Щ
simple_rnn_cell_8/MatMul_1MatMulzeros:output:01simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ъ
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ k
simple_rnn_cell_8/ReluRelusimple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource2simple_rnn_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1738535*
condR
while_cond_1738534*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype0h
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
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ ѕ
NoOpNoOp)^simple_rnn_cell_8/BiasAdd/ReadVariableOp(^simple_rnn_cell_8/MatMul/ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2T
(simple_rnn_cell_8/BiasAdd/ReadVariableOp(simple_rnn_cell_8/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_8/MatMul/ReadVariableOp'simple_rnn_cell_8/MatMul/ReadVariableOp2V
)simple_rnn_cell_8/MatMul_1/ReadVariableOp)simple_rnn_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
я
ѓ
while_cond_1736693
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1736693___redundant_placeholder05
1while_while_cond_1736693___redundant_placeholder15
1while_while_cond_1736693___redundant_placeholder25
1while_while_cond_1736693___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Щ
м
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_1740404

inputs
states_00
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€G
ReluReluadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states/0
К
є
/__inference_simple_rnn_10_layer_call_fn_1739239

inputs
unknown:
	unknown_0:
	unknown_1:
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1736347s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
 =
√
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1736347

inputsC
1simple_rnn_cell_10_matmul_readvariableop_resource:@
2simple_rnn_cell_10_biasadd_readvariableop_resource:E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:
identityИҐ)simple_rnn_cell_10/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_10/MatMul/ReadVariableOpҐ*simple_rnn_cell_10/MatMul_1/ReadVariableOpҐwhile;
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
value	B :s
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
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
valueB"€€€€   а
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
:€€€€€€€€€*
shrink_axis_maskЪ
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype0°
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Ы
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€m
simple_rnn_cell_10/ReluRelusimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1736281*
condR
while_cond_1736280*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0h
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
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€“
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
К
є
/__inference_simple_rnn_10_layer_call_fn_1739250

inputs
unknown:
	unknown_0:
	unknown_1:
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1736760s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Щ&
с
J__inference_sequential_19_layer_call_and_return_conditional_losses_1737210
simple_rnn_8_input&
simple_rnn_8_1737173: "
simple_rnn_8_1737175: &
simple_rnn_8_1737177:  &
simple_rnn_9_1737180: "
simple_rnn_9_1737182:&
simple_rnn_9_1737184:'
simple_rnn_10_1737188:#
simple_rnn_10_1737190:'
simple_rnn_10_1737192:'
simple_rnn_11_1737195: #
simple_rnn_11_1737197: '
simple_rnn_11_1737199:  -
time_distributed_18_1737202: )
time_distributed_18_1737204:
identityИҐ%simple_rnn_10/StatefulPartitionedCallҐ%simple_rnn_11/StatefulPartitionedCallҐ$simple_rnn_8/StatefulPartitionedCallҐ$simple_rnn_9/StatefulPartitionedCallҐ+time_distributed_18/StatefulPartitionedCallЂ
$simple_rnn_8/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_8_inputsimple_rnn_8_1737173simple_rnn_8_1737175simple_rnn_8_1737177*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1736114¬
$simple_rnn_9/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_8/StatefulPartitionedCall:output:0simple_rnn_9_1737180simple_rnn_9_1737182simple_rnn_9_1737184*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1736231ф
 repeat_vector_18/PartitionedCallPartitionedCall-simple_rnn_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_repeat_vector_18_layer_call_and_return_conditional_losses_1735332«
%simple_rnn_10/StatefulPartitionedCallStatefulPartitionedCall)repeat_vector_18/PartitionedCall:output:0simple_rnn_10_1737188simple_rnn_10_1737190simple_rnn_10_1737192*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1736347ћ
%simple_rnn_11/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_10/StatefulPartitionedCall:output:0simple_rnn_11_1737195simple_rnn_11_1737197simple_rnn_11_1737199*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1736462Ћ
+time_distributed_18/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_11/StatefulPartitionedCall:output:0time_distributed_18_1737202time_distributed_18_1737204*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_time_distributed_18_layer_call_and_return_conditional_losses_1735954r
!time_distributed_18/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    і
time_distributed_18/ReshapeReshape.simple_rnn_11/StatefulPartitionedCall:output:0*time_distributed_18/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ З
IdentityIdentity4time_distributed_18/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Т
NoOpNoOp&^simple_rnn_10/StatefulPartitionedCall&^simple_rnn_11/StatefulPartitionedCall%^simple_rnn_8/StatefulPartitionedCall%^simple_rnn_9/StatefulPartitionedCall,^time_distributed_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 2N
%simple_rnn_10/StatefulPartitionedCall%simple_rnn_10/StatefulPartitionedCall2N
%simple_rnn_11/StatefulPartitionedCall%simple_rnn_11/StatefulPartitionedCall2L
$simple_rnn_8/StatefulPartitionedCall$simple_rnn_8/StatefulPartitionedCall2L
$simple_rnn_9/StatefulPartitionedCall$simple_rnn_9/StatefulPartitionedCall2Z
+time_distributed_18/StatefulPartitionedCall+time_distributed_18/StatefulPartitionedCall:_ [
+
_output_shapes
:€€€€€€€€€
,
_user_specified_namesimple_rnn_8_input
я
ѓ
while_cond_1736280
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1736280___redundant_placeholder05
1while_while_cond_1736280___redundant_placeholder15
1while_while_cond_1736280___redundant_placeholder25
1while_while_cond_1736280___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
бE
ч
-sequential_19_simple_rnn_8_while_body_1734336R
Nsequential_19_simple_rnn_8_while_sequential_19_simple_rnn_8_while_loop_counterX
Tsequential_19_simple_rnn_8_while_sequential_19_simple_rnn_8_while_maximum_iterations0
,sequential_19_simple_rnn_8_while_placeholder2
.sequential_19_simple_rnn_8_while_placeholder_12
.sequential_19_simple_rnn_8_while_placeholder_2Q
Msequential_19_simple_rnn_8_while_sequential_19_simple_rnn_8_strided_slice_1_0О
Йsequential_19_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0e
Ssequential_19_simple_rnn_8_while_simple_rnn_cell_8_matmul_readvariableop_resource_0: b
Tsequential_19_simple_rnn_8_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0: g
Usequential_19_simple_rnn_8_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0:  -
)sequential_19_simple_rnn_8_while_identity/
+sequential_19_simple_rnn_8_while_identity_1/
+sequential_19_simple_rnn_8_while_identity_2/
+sequential_19_simple_rnn_8_while_identity_3/
+sequential_19_simple_rnn_8_while_identity_4O
Ksequential_19_simple_rnn_8_while_sequential_19_simple_rnn_8_strided_slice_1М
Зsequential_19_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorc
Qsequential_19_simple_rnn_8_while_simple_rnn_cell_8_matmul_readvariableop_resource: `
Rsequential_19_simple_rnn_8_while_simple_rnn_cell_8_biasadd_readvariableop_resource: e
Ssequential_19_simple_rnn_8_while_simple_rnn_cell_8_matmul_1_readvariableop_resource:  ИҐIsequential_19/simple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpҐHsequential_19/simple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOpҐJsequential_19/simple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp£
Rsequential_19/simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ѓ
Dsequential_19/simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЙsequential_19_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0,sequential_19_simple_rnn_8_while_placeholder[sequential_19/simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0№
Hsequential_19/simple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOpSsequential_19_simple_rnn_8_while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0Ф
9sequential_19/simple_rnn_8/while/simple_rnn_cell_8/MatMulMatMulKsequential_19/simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Psequential_19/simple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Џ
Isequential_19/simple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOpTsequential_19_simple_rnn_8_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0П
:sequential_19/simple_rnn_8/while/simple_rnn_cell_8/BiasAddBiasAddCsequential_19/simple_rnn_8/while/simple_rnn_cell_8/MatMul:product:0Qsequential_19/simple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ а
Jsequential_19/simple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOpUsequential_19_simple_rnn_8_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0ы
;sequential_19/simple_rnn_8/while/simple_rnn_cell_8/MatMul_1MatMul.sequential_19_simple_rnn_8_while_placeholder_2Rsequential_19/simple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ э
6sequential_19/simple_rnn_8/while/simple_rnn_cell_8/addAddV2Csequential_19/simple_rnn_8/while/simple_rnn_cell_8/BiasAdd:output:0Esequential_19/simple_rnn_8/while/simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ ≠
7sequential_19/simple_rnn_8/while/simple_rnn_cell_8/ReluRelu:sequential_19/simple_rnn_8/while/simple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ њ
Esequential_19/simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_19_simple_rnn_8_while_placeholder_1,sequential_19_simple_rnn_8_while_placeholderEsequential_19/simple_rnn_8/while/simple_rnn_cell_8/Relu:activations:0*
_output_shapes
: *
element_dtype0:йи“h
&sequential_19/simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :≠
$sequential_19/simple_rnn_8/while/addAddV2,sequential_19_simple_rnn_8_while_placeholder/sequential_19/simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: j
(sequential_19/simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :”
&sequential_19/simple_rnn_8/while/add_1AddV2Nsequential_19_simple_rnn_8_while_sequential_19_simple_rnn_8_while_loop_counter1sequential_19/simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: ™
)sequential_19/simple_rnn_8/while/IdentityIdentity*sequential_19/simple_rnn_8/while/add_1:z:0&^sequential_19/simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ÷
+sequential_19/simple_rnn_8/while/Identity_1IdentityTsequential_19_simple_rnn_8_while_sequential_19_simple_rnn_8_while_maximum_iterations&^sequential_19/simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ™
+sequential_19/simple_rnn_8/while/Identity_2Identity(sequential_19/simple_rnn_8/while/add:z:0&^sequential_19/simple_rnn_8/while/NoOp*
T0*
_output_shapes
: „
+sequential_19/simple_rnn_8/while/Identity_3IdentityUsequential_19/simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^sequential_19/simple_rnn_8/while/NoOp*
T0*
_output_shapes
: Ў
+sequential_19/simple_rnn_8/while/Identity_4IdentityEsequential_19/simple_rnn_8/while/simple_rnn_cell_8/Relu:activations:0&^sequential_19/simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Ћ
%sequential_19/simple_rnn_8/while/NoOpNoOpJ^sequential_19/simple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpI^sequential_19/simple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOpK^sequential_19/simple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "_
)sequential_19_simple_rnn_8_while_identity2sequential_19/simple_rnn_8/while/Identity:output:0"c
+sequential_19_simple_rnn_8_while_identity_14sequential_19/simple_rnn_8/while/Identity_1:output:0"c
+sequential_19_simple_rnn_8_while_identity_24sequential_19/simple_rnn_8/while/Identity_2:output:0"c
+sequential_19_simple_rnn_8_while_identity_34sequential_19/simple_rnn_8/while/Identity_3:output:0"c
+sequential_19_simple_rnn_8_while_identity_44sequential_19/simple_rnn_8/while/Identity_4:output:0"Ь
Ksequential_19_simple_rnn_8_while_sequential_19_simple_rnn_8_strided_slice_1Msequential_19_simple_rnn_8_while_sequential_19_simple_rnn_8_strided_slice_1_0"™
Rsequential_19_simple_rnn_8_while_simple_rnn_cell_8_biasadd_readvariableop_resourceTsequential_19_simple_rnn_8_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"ђ
Ssequential_19_simple_rnn_8_while_simple_rnn_cell_8_matmul_1_readvariableop_resourceUsequential_19_simple_rnn_8_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"®
Qsequential_19_simple_rnn_8_while_simple_rnn_cell_8_matmul_readvariableop_resourceSsequential_19_simple_rnn_8_while_simple_rnn_cell_8_matmul_readvariableop_resource_0"Ц
Зsequential_19_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorЙsequential_19_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2Ц
Isequential_19/simple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpIsequential_19/simple_rnn_8/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2Ф
Hsequential_19/simple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOpHsequential_19/simple_rnn_8/while/simple_rnn_cell_8/MatMul/ReadVariableOp2Ш
Jsequential_19/simple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpJsequential_19/simple_rnn_8/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
№-
…
while_body_1738906
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_9_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_9_matmul_readvariableop_resource: E
7while_simple_rnn_cell_9_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:ИҐ.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_9/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0¶
-while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0√
while/simple_rnn_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€§
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Њ
while/simple_rnn_cell_9/BiasAddBiasAdd(while/simple_rnn_cell_9/MatMul:product:06while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€™
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0™
 while/simple_rnn_cell_9/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
while/simple_rnn_cell_9/addAddV2(while/simple_rnn_cell_9/BiasAdd:output:0*while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€w
while/simple_rnn_cell_9/ReluReluwhile/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ы
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_9/Relu:activations:0*
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
: З
while/Identity_4Identity*while/simple_rnn_cell_9/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€я

while/NoOpNoOp/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_9_biasadd_readvariableop_resource9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_9_matmul_readvariableop_resource8while_simple_rnn_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2`
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_9/MatMul/ReadVariableOp-while/simple_rnn_cell_9/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
И>
≈
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1739466
inputs_0C
1simple_rnn_cell_10_matmul_readvariableop_resource:@
2simple_rnn_cell_10_biasadd_readvariableop_resource:E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:
identityИҐ)simple_rnn_cell_10/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_10/MatMul/ReadVariableOpҐ*simple_rnn_cell_10/MatMul_1/ReadVariableOpҐwhile=
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
value	B :s
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
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
valueB"€€€€   а
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
:€€€€€€€€€*
shrink_axis_maskЪ
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype0°
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Ы
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€m
simple_rnn_cell_10/ReluRelusimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1739400*
condR
while_cond_1739399*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
element_dtype0h
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
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€“
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
я
ѓ
while_cond_1738905
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1738905___redundant_placeholder05
1while_while_cond_1738905___redundant_placeholder15
1while_while_cond_1738905___redundant_placeholder25
1while_while_cond_1738905___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
љ

№
4__inference_simple_rnn_cell_10_layer_call_fn_1740356

inputs
states_0
unknown:
	unknown_0:
	unknown_1:
identity

identity_1ИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_1735383o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states/0
ї

џ
3__inference_simple_rnn_cell_8_layer_call_fn_1740246

inputs
states_0
unknown: 
	unknown_0: 
	unknown_1:  
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
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_1734900o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/0
я
ѓ
while_cond_1738426
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1738426___redundant_placeholder05
1while_while_cond_1738426___redundant_placeholder15
1while_while_cond_1738426___redundant_placeholder25
1while_while_cond_1738426___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Ђ>
Љ
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1739193

inputsB
0simple_rnn_cell_9_matmul_readvariableop_resource: ?
1simple_rnn_cell_9_biasadd_readvariableop_resource:D
2simple_rnn_cell_9_matmul_1_readvariableop_resource:
identityИҐ(simple_rnn_cell_9/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_9/MatMul/ReadVariableOpҐ)simple_rnn_cell_9/MatMul_1/ReadVariableOpҐwhile;
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
value	B :s
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ D
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
valueB"€€€€    а
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
:€€€€€€€€€ *
shrink_axis_maskШ
'simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Я
simple_rnn_cell_9/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ц
(simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ђ
simple_rnn_cell_9/BiasAddBiasAdd"simple_rnn_cell_9/MatMul:product:00simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
)simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Щ
simple_rnn_cell_9/MatMul_1MatMulzeros:output:01simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
simple_rnn_cell_9/addAddV2"simple_rnn_cell_9/BiasAdd:output:0$simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€k
simple_rnn_cell_9/ReluRelusimple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_9_matmul_readvariableop_resource1simple_rnn_cell_9_biasadd_readvariableop_resource2simple_rnn_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1739126*
condR
while_cond_1739125*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
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
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ѕ
NoOpNoOp)^simple_rnn_cell_9/BiasAdd/ReadVariableOp(^simple_rnn_cell_9/MatMul/ReadVariableOp*^simple_rnn_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ : : : 2T
(simple_rnn_cell_9/BiasAdd/ReadVariableOp(simple_rnn_cell_9/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_9/MatMul/ReadVariableOp'simple_rnn_cell_9/MatMul/ReadVariableOp2V
)simple_rnn_cell_9/MatMul_1/ReadVariableOp)simple_rnn_cell_9/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
И>
≈
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1739942
inputs_0C
1simple_rnn_cell_11_matmul_readvariableop_resource: @
2simple_rnn_cell_11_biasadd_readvariableop_resource: E
3simple_rnn_cell_11_matmul_1_readvariableop_resource:  
identityИҐ)simple_rnn_cell_11/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_11/MatMul/ReadVariableOpҐ*simple_rnn_cell_11/MatMul_1/ReadVariableOpҐwhile=
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
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
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
valueB"€€€€   а
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
:€€€€€€€€€*
shrink_axis_maskЪ
(simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

: *
dtype0°
simple_rnn_cell_11/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ш
)simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ѓ
simple_rnn_cell_11/BiasAddBiasAdd#simple_rnn_cell_11/MatMul:product:01simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ю
*simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0Ы
simple_rnn_cell_11/MatMul_1MatMulzeros:output:02simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Э
simple_rnn_cell_11/addAddV2#simple_rnn_cell_11/BiasAdd:output:0%simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ m
simple_rnn_cell_11/ReluRelusimple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_11_matmul_readvariableop_resource2simple_rnn_cell_11_biasadd_readvariableop_resource3simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1739876*
condR
while_cond_1739875*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
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
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ “
NoOpNoOp*^simple_rnn_cell_11/BiasAdd/ReadVariableOp)^simple_rnn_cell_11/MatMul/ReadVariableOp+^simple_rnn_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2V
)simple_rnn_cell_11/BiasAdd/ReadVariableOp)simple_rnn_cell_11/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_11/MatMul/ReadVariableOp(simple_rnn_cell_11/MatMul/ReadVariableOp2X
*simple_rnn_cell_11/MatMul_1/ReadVariableOp*simple_rnn_cell_11/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
»	
ц
E__inference_dense_21_layer_call_and_return_conditional_losses_1740485

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
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
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
я
ѓ
while_cond_1738642
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1738642___redundant_placeholder05
1while_while_cond_1738642___redundant_placeholder15
1while_while_cond_1738642___redundant_placeholder25
1while_while_cond_1738642___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
ј,
…
while_body_1738319
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0: L
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_8_matmul_readvariableop_resource: E
7while_simple_rnn_cell_8_biasadd_readvariableop_resource: J
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:  ИҐ.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_8/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0¶
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0√
while/simple_rnn_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ §
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Њ
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ™
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0™
 while/simple_rnn_cell_8/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ђ
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ w
while/simple_rnn_cell_8/ReluReluwhile/simple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ”
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_8/Relu:activations:0*
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
: З
while/Identity_4Identity*while/simple_rnn_cell_8/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ я

while/NoOpNoOp/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2`
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_8/MatMul/ReadVariableOp-while/simple_rnn_cell_8/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
я
ѓ
while_cond_1735687
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1735687___redundant_placeholder05
1while_while_cond_1735687___redundant_placeholder15
1while_while_cond_1735687___redundant_placeholder25
1while_while_cond_1735687___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
я
ѓ
while_cond_1738795
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1738795___redundant_placeholder05
1while_while_cond_1738795___redundant_placeholder15
1while_while_cond_1738795___redundant_placeholder25
1while_while_cond_1738795___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
У
к
O__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_1735675

inputs

states0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 2
 matmul_1_readvariableop_resource:  
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ G
ReluReluadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
я
ѓ
while_cond_1739125
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1739125___redundant_placeholder05
1while_while_cond_1739125___redundant_placeholder15
1while_while_cond_1739125___redundant_placeholder25
1while_while_cond_1739125___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
я
ѓ
while_cond_1739983
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1739983___redundant_placeholder05
1while_while_cond_1739983___redundant_placeholder15
1while_while_cond_1739983___redundant_placeholder25
1while_while_cond_1739983___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
м,
“
while_body_1736564
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_11_matmul_readvariableop_resource_0: H
:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0: M
;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_11_matmul_readvariableop_resource: F
8while_simple_rnn_cell_11_biasadd_readvariableop_resource: K
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource:  ИҐ/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_11/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0®
.while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0≈
while/simple_rnn_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ¶
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ѕ
 while/simple_rnn_cell_11/BiasAddBiasAdd)while/simple_rnn_cell_11/MatMul:product:07while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ђ
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0ђ
!while/simple_rnn_cell_11/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ѓ
while/simple_rnn_cell_11/addAddV2)while/simple_rnn_cell_11/BiasAdd:output:0+while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ y
while/simple_rnn_cell_11/ReluRelu while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ‘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder+while/simple_rnn_cell_11/Relu:activations:0*
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
: И
while/Identity_4Identity+while/simple_rnn_cell_11/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ в

while/NoOpNoOp0^while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_11/MatMul/ReadVariableOp1^while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_11_matmul_readvariableop_resource9while_simple_rnn_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2b
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_11/MatMul/ReadVariableOp.while/simple_rnn_cell_11/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
’
√
.sequential_19_simple_rnn_11_while_cond_1734653T
Psequential_19_simple_rnn_11_while_sequential_19_simple_rnn_11_while_loop_counterZ
Vsequential_19_simple_rnn_11_while_sequential_19_simple_rnn_11_while_maximum_iterations1
-sequential_19_simple_rnn_11_while_placeholder3
/sequential_19_simple_rnn_11_while_placeholder_13
/sequential_19_simple_rnn_11_while_placeholder_2V
Rsequential_19_simple_rnn_11_while_less_sequential_19_simple_rnn_11_strided_slice_1m
isequential_19_simple_rnn_11_while_sequential_19_simple_rnn_11_while_cond_1734653___redundant_placeholder0m
isequential_19_simple_rnn_11_while_sequential_19_simple_rnn_11_while_cond_1734653___redundant_placeholder1m
isequential_19_simple_rnn_11_while_sequential_19_simple_rnn_11_while_cond_1734653___redundant_placeholder2m
isequential_19_simple_rnn_11_while_sequential_19_simple_rnn_11_while_cond_1734653___redundant_placeholder3.
*sequential_19_simple_rnn_11_while_identity
“
&sequential_19/simple_rnn_11/while/LessLess-sequential_19_simple_rnn_11_while_placeholderRsequential_19_simple_rnn_11_while_less_sequential_19_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: Г
*sequential_19/simple_rnn_11/while/IdentityIdentity*sequential_19/simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "a
*sequential_19_simple_rnn_11_while_identity3sequential_19/simple_rnn_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
я
ѓ
while_cond_1735395
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1735395___redundant_placeholder05
1while_while_cond_1735395___redundant_placeholder15
1while_while_cond_1735395___redundant_placeholder25
1while_while_cond_1735395___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
г=
Њ
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1738493
inputs_0B
0simple_rnn_cell_8_matmul_readvariableop_resource: ?
1simple_rnn_cell_8_biasadd_readvariableop_resource: D
2simple_rnn_cell_8_matmul_1_readvariableop_resource:  
identityИҐ(simple_rnn_cell_8/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_8/MatMul/ReadVariableOpҐ)simple_rnn_cell_8/MatMul_1/ReadVariableOpҐwhile=
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
:€€€€€€€€€ c
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
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Я
simple_rnn_cell_8/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ц
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ђ
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ь
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0Щ
simple_rnn_cell_8/MatMul_1MatMulzeros:output:01simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ъ
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ k
simple_rnn_cell_8/ReluRelusimple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource2simple_rnn_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1738427*
condR
while_cond_1738426*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
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
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ ѕ
NoOpNoOp)^simple_rnn_cell_8/BiasAdd/ReadVariableOp(^simple_rnn_cell_8/MatMul/ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2T
(simple_rnn_cell_8/BiasAdd/ReadVariableOp(simple_rnn_cell_8/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_8/MatMul/ReadVariableOp'simple_rnn_cell_8/MatMul/ReadVariableOp2V
)simple_rnn_cell_8/MatMul_1/ReadVariableOp)simple_rnn_cell_8/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
ДG
Ю
.sequential_19_simple_rnn_11_while_body_1734654T
Psequential_19_simple_rnn_11_while_sequential_19_simple_rnn_11_while_loop_counterZ
Vsequential_19_simple_rnn_11_while_sequential_19_simple_rnn_11_while_maximum_iterations1
-sequential_19_simple_rnn_11_while_placeholder3
/sequential_19_simple_rnn_11_while_placeholder_13
/sequential_19_simple_rnn_11_while_placeholder_2S
Osequential_19_simple_rnn_11_while_sequential_19_simple_rnn_11_strided_slice_1_0Р
Лsequential_19_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0g
Usequential_19_simple_rnn_11_while_simple_rnn_cell_11_matmul_readvariableop_resource_0: d
Vsequential_19_simple_rnn_11_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0: i
Wsequential_19_simple_rnn_11_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:  .
*sequential_19_simple_rnn_11_while_identity0
,sequential_19_simple_rnn_11_while_identity_10
,sequential_19_simple_rnn_11_while_identity_20
,sequential_19_simple_rnn_11_while_identity_30
,sequential_19_simple_rnn_11_while_identity_4Q
Msequential_19_simple_rnn_11_while_sequential_19_simple_rnn_11_strided_slice_1О
Йsequential_19_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_11_tensorarrayunstack_tensorlistfromtensore
Ssequential_19_simple_rnn_11_while_simple_rnn_cell_11_matmul_readvariableop_resource: b
Tsequential_19_simple_rnn_11_while_simple_rnn_cell_11_biasadd_readvariableop_resource: g
Usequential_19_simple_rnn_11_while_simple_rnn_cell_11_matmul_1_readvariableop_resource:  ИҐKsequential_19/simple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpҐJsequential_19/simple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOpҐLsequential_19/simple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp§
Ssequential_19/simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ≥
Esequential_19/simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЛsequential_19_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0-sequential_19_simple_rnn_11_while_placeholder\sequential_19/simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0а
Jsequential_19/simple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOpUsequential_19_simple_rnn_11_while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0Щ
;sequential_19/simple_rnn_11/while/simple_rnn_cell_11/MatMulMatMulLsequential_19/simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Rsequential_19/simple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ё
Ksequential_19/simple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOpVsequential_19_simple_rnn_11_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Х
<sequential_19/simple_rnn_11/while/simple_rnn_cell_11/BiasAddBiasAddEsequential_19/simple_rnn_11/while/simple_rnn_cell_11/MatMul:product:0Ssequential_19/simple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ д
Lsequential_19/simple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOpWsequential_19_simple_rnn_11_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0А
=sequential_19/simple_rnn_11/while/simple_rnn_cell_11/MatMul_1MatMul/sequential_19_simple_rnn_11_while_placeholder_2Tsequential_19/simple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Г
8sequential_19/simple_rnn_11/while/simple_rnn_cell_11/addAddV2Esequential_19/simple_rnn_11/while/simple_rnn_cell_11/BiasAdd:output:0Gsequential_19/simple_rnn_11/while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ ±
9sequential_19/simple_rnn_11/while/simple_rnn_cell_11/ReluRelu<sequential_19/simple_rnn_11/while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ƒ
Fsequential_19/simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem/sequential_19_simple_rnn_11_while_placeholder_1-sequential_19_simple_rnn_11_while_placeholderGsequential_19/simple_rnn_11/while/simple_rnn_cell_11/Relu:activations:0*
_output_shapes
: *
element_dtype0:йи“i
'sequential_19/simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :∞
%sequential_19/simple_rnn_11/while/addAddV2-sequential_19_simple_rnn_11_while_placeholder0sequential_19/simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: k
)sequential_19/simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :„
'sequential_19/simple_rnn_11/while/add_1AddV2Psequential_19_simple_rnn_11_while_sequential_19_simple_rnn_11_while_loop_counter2sequential_19/simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: ≠
*sequential_19/simple_rnn_11/while/IdentityIdentity+sequential_19/simple_rnn_11/while/add_1:z:0'^sequential_19/simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Џ
,sequential_19/simple_rnn_11/while/Identity_1IdentityVsequential_19_simple_rnn_11_while_sequential_19_simple_rnn_11_while_maximum_iterations'^sequential_19/simple_rnn_11/while/NoOp*
T0*
_output_shapes
: ≠
,sequential_19/simple_rnn_11/while/Identity_2Identity)sequential_19/simple_rnn_11/while/add:z:0'^sequential_19/simple_rnn_11/while/NoOp*
T0*
_output_shapes
: Џ
,sequential_19/simple_rnn_11/while/Identity_3IdentityVsequential_19/simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^sequential_19/simple_rnn_11/while/NoOp*
T0*
_output_shapes
: №
,sequential_19/simple_rnn_11/while/Identity_4IdentityGsequential_19/simple_rnn_11/while/simple_rnn_cell_11/Relu:activations:0'^sequential_19/simple_rnn_11/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ “
&sequential_19/simple_rnn_11/while/NoOpNoOpL^sequential_19/simple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpK^sequential_19/simple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOpM^sequential_19/simple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "a
*sequential_19_simple_rnn_11_while_identity3sequential_19/simple_rnn_11/while/Identity:output:0"e
,sequential_19_simple_rnn_11_while_identity_15sequential_19/simple_rnn_11/while/Identity_1:output:0"e
,sequential_19_simple_rnn_11_while_identity_25sequential_19/simple_rnn_11/while/Identity_2:output:0"e
,sequential_19_simple_rnn_11_while_identity_35sequential_19/simple_rnn_11/while/Identity_3:output:0"e
,sequential_19_simple_rnn_11_while_identity_45sequential_19/simple_rnn_11/while/Identity_4:output:0"†
Msequential_19_simple_rnn_11_while_sequential_19_simple_rnn_11_strided_slice_1Osequential_19_simple_rnn_11_while_sequential_19_simple_rnn_11_strided_slice_1_0"Ѓ
Tsequential_19_simple_rnn_11_while_simple_rnn_cell_11_biasadd_readvariableop_resourceVsequential_19_simple_rnn_11_while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"∞
Usequential_19_simple_rnn_11_while_simple_rnn_cell_11_matmul_1_readvariableop_resourceWsequential_19_simple_rnn_11_while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"ђ
Ssequential_19_simple_rnn_11_while_simple_rnn_cell_11_matmul_readvariableop_resourceUsequential_19_simple_rnn_11_while_simple_rnn_cell_11_matmul_readvariableop_resource_0"Ъ
Йsequential_19_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorЛsequential_19_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2Ъ
Ksequential_19/simple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpKsequential_19/simple_rnn_11/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2Ш
Jsequential_19/simple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOpJsequential_19/simple_rnn_11/while/simple_rnn_cell_11/MatMul/ReadVariableOp2Ь
Lsequential_19/simple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOpLsequential_19/simple_rnn_11/while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Ю!
я
while_body_1735555
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
"while_simple_rnn_cell_10_1735577_0:0
"while_simple_rnn_cell_10_1735579_0:4
"while_simple_rnn_cell_10_1735581_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
 while_simple_rnn_cell_10_1735577:.
 while_simple_rnn_cell_10_1735579:2
 while_simple_rnn_cell_10_1735581:ИҐ0while/simple_rnn_cell_10/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Ђ
0while/simple_rnn_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2"while_simple_rnn_cell_10_1735577_0"while_simple_rnn_cell_10_1735579_0"while_simple_rnn_cell_10_1735581_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_1735503в
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_10/StatefulPartitionedCall:output:0*
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
: Ц
while/Identity_4Identity9while/simple_rnn_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€

while/NoOpNoOp1^while/simple_rnn_cell_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"F
 while_simple_rnn_cell_10_1735577"while_simple_rnn_cell_10_1735577_0"F
 while_simple_rnn_cell_10_1735579"while_simple_rnn_cell_10_1735579_0"F
 while_simple_rnn_cell_10_1735581"while_simple_rnn_cell_10_1735581_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2d
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
№-
…
while_body_1736825
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_9_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_9_matmul_readvariableop_resource: E
7while_simple_rnn_cell_9_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:ИҐ.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_9/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0¶
-while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0√
while/simple_rnn_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€§
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Њ
while/simple_rnn_cell_9/BiasAddBiasAdd(while/simple_rnn_cell_9/MatMul:product:06while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€™
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0™
 while/simple_rnn_cell_9/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
while/simple_rnn_cell_9/addAddV2(while/simple_rnn_cell_9/BiasAdd:output:0*while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€w
while/simple_rnn_cell_9/ReluReluwhile/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ы
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_9/Relu:activations:0*
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
: З
while/Identity_4Identity*while/simple_rnn_cell_9/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€я

while/NoOpNoOp/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_9_biasadd_readvariableop_resource9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_9_matmul_readvariableop_resource8while_simple_rnn_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2`
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_9/MatMul/ReadVariableOp-while/simple_rnn_cell_9/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
ј,
…
while_body_1738427
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0: L
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_8_matmul_readvariableop_resource: E
7while_simple_rnn_cell_8_biasadd_readvariableop_resource: J
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:  ИҐ.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_8/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0¶
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0√
while/simple_rnn_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ §
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Њ
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ™
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0™
 while/simple_rnn_cell_8/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ђ
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ w
while/simple_rnn_cell_8/ReluReluwhile/simple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ”
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_8/Relu:activations:0*
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
: З
while/Identity_4Identity*while/simple_rnn_cell_8/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ я

while/NoOpNoOp/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2`
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_8/MatMul/ReadVariableOp-while/simple_rnn_cell_8/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Щ&
с
J__inference_sequential_19_layer_call_and_return_conditional_losses_1737250
simple_rnn_8_input&
simple_rnn_8_1737213: "
simple_rnn_8_1737215: &
simple_rnn_8_1737217:  &
simple_rnn_9_1737220: "
simple_rnn_9_1737222:&
simple_rnn_9_1737224:'
simple_rnn_10_1737228:#
simple_rnn_10_1737230:'
simple_rnn_10_1737232:'
simple_rnn_11_1737235: #
simple_rnn_11_1737237: '
simple_rnn_11_1737239:  -
time_distributed_18_1737242: )
time_distributed_18_1737244:
identityИҐ%simple_rnn_10/StatefulPartitionedCallҐ%simple_rnn_11/StatefulPartitionedCallҐ$simple_rnn_8/StatefulPartitionedCallҐ$simple_rnn_9/StatefulPartitionedCallҐ+time_distributed_18/StatefulPartitionedCallЂ
$simple_rnn_8/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_8_inputsimple_rnn_8_1737213simple_rnn_8_1737215simple_rnn_8_1737217*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1737022¬
$simple_rnn_9/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_8/StatefulPartitionedCall:output:0simple_rnn_9_1737220simple_rnn_9_1737222simple_rnn_9_1737224*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1736892ф
 repeat_vector_18/PartitionedCallPartitionedCall-simple_rnn_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_repeat_vector_18_layer_call_and_return_conditional_losses_1735332«
%simple_rnn_10/StatefulPartitionedCallStatefulPartitionedCall)repeat_vector_18/PartitionedCall:output:0simple_rnn_10_1737228simple_rnn_10_1737230simple_rnn_10_1737232*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1736760ћ
%simple_rnn_11/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_10/StatefulPartitionedCall:output:0simple_rnn_11_1737235simple_rnn_11_1737237simple_rnn_11_1737239*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1736630Ћ
+time_distributed_18/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_11/StatefulPartitionedCall:output:0time_distributed_18_1737242time_distributed_18_1737244*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_time_distributed_18_layer_call_and_return_conditional_losses_1735993r
!time_distributed_18/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    і
time_distributed_18/ReshapeReshape.simple_rnn_11/StatefulPartitionedCall:output:0*time_distributed_18/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ З
IdentityIdentity4time_distributed_18/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Т
NoOpNoOp&^simple_rnn_10/StatefulPartitionedCall&^simple_rnn_11/StatefulPartitionedCall%^simple_rnn_8/StatefulPartitionedCall%^simple_rnn_9/StatefulPartitionedCall,^time_distributed_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 2N
%simple_rnn_10/StatefulPartitionedCall%simple_rnn_10/StatefulPartitionedCall2N
%simple_rnn_11/StatefulPartitionedCall%simple_rnn_11/StatefulPartitionedCall2L
$simple_rnn_8/StatefulPartitionedCall$simple_rnn_8/StatefulPartitionedCall2L
$simple_rnn_9/StatefulPartitionedCall$simple_rnn_9/StatefulPartitionedCall2Z
+time_distributed_18/StatefulPartitionedCall+time_distributed_18/StatefulPartitionedCall:_ [
+
_output_shapes
:€€€€€€€€€
,
_user_specified_namesimple_rnn_8_input
я
ѓ
while_cond_1734951
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1734951___redundant_placeholder05
1while_while_cond_1734951___redundant_placeholder15
1while_while_cond_1734951___redundant_placeholder25
1while_while_cond_1734951___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Т
й
N__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_1735072

inputs

states0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€G
ReluReluadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€ :€€€€€€€€€: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namestates
м,
“
while_body_1740092
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_11_matmul_readvariableop_resource_0: H
:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0: M
;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_11_matmul_readvariableop_resource: F
8while_simple_rnn_cell_11_biasadd_readvariableop_resource: K
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource:  ИҐ/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_11/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0®
.while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0≈
while/simple_rnn_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ¶
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ѕ
 while/simple_rnn_cell_11/BiasAddBiasAdd)while/simple_rnn_cell_11/MatMul:product:07while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ђ
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0ђ
!while/simple_rnn_cell_11/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ѓ
while/simple_rnn_cell_11/addAddV2)while/simple_rnn_cell_11/BiasAdd:output:0+while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ y
while/simple_rnn_cell_11/ReluRelu while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ‘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder+while/simple_rnn_cell_11/Relu:activations:0*
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
: И
while/Identity_4Identity+while/simple_rnn_cell_11/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ в

while/NoOpNoOp0^while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_11/MatMul/ReadVariableOp1^while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_11_matmul_readvariableop_resource9while_simple_rnn_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2b
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_11/MatMul/ReadVariableOp.while/simple_rnn_cell_11/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
ј,
…
while_body_1736048
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0: L
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_8_matmul_readvariableop_resource: E
7while_simple_rnn_cell_8_biasadd_readvariableop_resource: J
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:  ИҐ.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_8/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0¶
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0√
while/simple_rnn_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ §
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Њ
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ™
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0™
 while/simple_rnn_cell_8/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ђ
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ w
while/simple_rnn_cell_8/ReluReluwhile/simple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ”
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_8/Relu:activations:0*
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
: З
while/Identity_4Identity*while/simple_rnn_cell_8/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ я

while/NoOpNoOp/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2`
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_8/MatMul/ReadVariableOp-while/simple_rnn_cell_8/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
№-
…
while_body_1739126
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_9_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_9_matmul_readvariableop_resource: E
7while_simple_rnn_cell_9_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:ИҐ.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_9/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0¶
-while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0√
while/simple_rnn_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€§
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Њ
while/simple_rnn_cell_9/BiasAddBiasAdd(while/simple_rnn_cell_9/MatMul:product:06while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€™
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0™
 while/simple_rnn_cell_9/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
while/simple_rnn_cell_9/addAddV2(while/simple_rnn_cell_9/BiasAdd:output:0*while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€w
while/simple_rnn_cell_9/ReluReluwhile/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ы
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_9/Relu:activations:0*
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
: З
while/Identity_4Identity*while/simple_rnn_cell_9/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€я

while/NoOpNoOp/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_9_biasadd_readvariableop_resource9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_9_matmul_readvariableop_resource8while_simple_rnn_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2`
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_9/MatMul/ReadVariableOp-while/simple_rnn_cell_9/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
£"
Ў
while_body_1735086
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
!while_simple_rnn_cell_9_1735108_0: /
!while_simple_rnn_cell_9_1735110_0:3
!while_simple_rnn_cell_9_1735112_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
while_simple_rnn_cell_9_1735108: -
while_simple_rnn_cell_9_1735110:1
while_simple_rnn_cell_9_1735112:ИҐ/while/simple_rnn_cell_9/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0¶
/while/simple_rnn_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_9_1735108_0!while_simple_rnn_cell_9_1735110_0!while_simple_rnn_cell_9_1735112_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_1735072r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Й
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:08while/simple_rnn_cell_9/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity8while/simple_rnn_cell_9/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€~

while/NoOpNoOp0^while/simple_rnn_cell_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_9_1735108!while_simple_rnn_cell_9_1735108_0"D
while_simple_rnn_cell_9_1735110!while_simple_rnn_cell_9_1735110_0"D
while_simple_rnn_cell_9_1735112!while_simple_rnn_cell_9_1735112_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2b
/while/simple_rnn_cell_9/StatefulPartitionedCall/while/simple_rnn_cell_9/StatefulPartitionedCall: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
э9
ѕ
simple_rnn_9_while_body_17375046
2simple_rnn_9_while_simple_rnn_9_while_loop_counter<
8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations"
simple_rnn_9_while_placeholder$
 simple_rnn_9_while_placeholder_1$
 simple_rnn_9_while_placeholder_25
1simple_rnn_9_while_simple_rnn_9_strided_slice_1_0q
msimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_9_while_simple_rnn_cell_9_matmul_readvariableop_resource_0: T
Fsimple_rnn_9_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0:Y
Gsimple_rnn_9_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0:
simple_rnn_9_while_identity!
simple_rnn_9_while_identity_1!
simple_rnn_9_while_identity_2!
simple_rnn_9_while_identity_3!
simple_rnn_9_while_identity_43
/simple_rnn_9_while_simple_rnn_9_strided_slice_1o
ksimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_9_while_simple_rnn_cell_9_matmul_readvariableop_resource: R
Dsimple_rnn_9_while_simple_rnn_cell_9_biasadd_readvariableop_resource:W
Esimple_rnn_9_while_simple_rnn_cell_9_matmul_1_readvariableop_resource:ИҐ;simple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOpҐ:simple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOpҐ<simple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpХ
Dsimple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    з
6simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_9_while_placeholderMsimple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0ј
:simple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_9_while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0к
+simple_rnn_9/while/simple_rnn_cell_9/MatMulMatMul=simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Њ
;simple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_9_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0е
,simple_rnn_9/while/simple_rnn_cell_9/BiasAddBiasAdd5simple_rnn_9/while/simple_rnn_cell_9/MatMul:product:0Csimple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ƒ
<simple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_9_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0—
-simple_rnn_9/while/simple_rnn_cell_9/MatMul_1MatMul simple_rnn_9_while_placeholder_2Dsimple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€”
(simple_rnn_9/while/simple_rnn_cell_9/addAddV25simple_rnn_9/while/simple_rnn_cell_9/BiasAdd:output:07simple_rnn_9/while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€С
)simple_rnn_9/while/simple_rnn_cell_9/ReluRelu,simple_rnn_9/while/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€
=simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ѓ
7simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_9_while_placeholder_1Fsimple_rnn_9/while/TensorArrayV2Write/TensorListSetItem/index:output:07simple_rnn_9/while/simple_rnn_cell_9/Relu:activations:0*
_output_shapes
: *
element_dtype0:йи“Z
simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Г
simple_rnn_9/while/addAddV2simple_rnn_9_while_placeholder!simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ы
simple_rnn_9/while/add_1AddV22simple_rnn_9_while_simple_rnn_9_while_loop_counter#simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: А
simple_rnn_9/while/IdentityIdentitysimple_rnn_9/while/add_1:z:0^simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Ю
simple_rnn_9/while/Identity_1Identity8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations^simple_rnn_9/while/NoOp*
T0*
_output_shapes
: А
simple_rnn_9/while/Identity_2Identitysimple_rnn_9/while/add:z:0^simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ≠
simple_rnn_9/while/Identity_3IdentityGsimple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_9/while/NoOp*
T0*
_output_shapes
: Ѓ
simple_rnn_9/while/Identity_4Identity7simple_rnn_9/while/simple_rnn_cell_9/Relu:activations:0^simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€У
simple_rnn_9/while/NoOpNoOp<^simple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp;^simple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOp=^simple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_9_while_identity$simple_rnn_9/while/Identity:output:0"G
simple_rnn_9_while_identity_1&simple_rnn_9/while/Identity_1:output:0"G
simple_rnn_9_while_identity_2&simple_rnn_9/while/Identity_2:output:0"G
simple_rnn_9_while_identity_3&simple_rnn_9/while/Identity_3:output:0"G
simple_rnn_9_while_identity_4&simple_rnn_9/while/Identity_4:output:0"d
/simple_rnn_9_while_simple_rnn_9_strided_slice_11simple_rnn_9_while_simple_rnn_9_strided_slice_1_0"О
Dsimple_rnn_9_while_simple_rnn_cell_9_biasadd_readvariableop_resourceFsimple_rnn_9_while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"Р
Esimple_rnn_9_while_simple_rnn_cell_9_matmul_1_readvariableop_resourceGsimple_rnn_9_while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"М
Csimple_rnn_9_while_simple_rnn_cell_9_matmul_readvariableop_resourceEsimple_rnn_9_while_simple_rnn_cell_9_matmul_readvariableop_resource_0"№
ksimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensormsimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2z
;simple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp;simple_rnn_9/while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2x
:simple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOp:simple_rnn_9/while/simple_rnn_cell_9/MatMul/ReadVariableOp2|
<simple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp<simple_rnn_9/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
я
ѓ
while_cond_1736047
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1736047___redundant_placeholder05
1while_while_cond_1736047___redundant_placeholder15
1while_while_cond_1736047___redundant_placeholder25
1while_while_cond_1736047___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
ї

џ
3__inference_simple_rnn_cell_9_layer_call_fn_1740294

inputs
states_0
unknown: 
	unknown_0:
	unknown_1:
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
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_1735072o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€ :€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states/0
Ш
л
N__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_1740342

inputs
states_00
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€G
ReluReluadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€ :€€€€€€€€€: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states/0
Ш
л
N__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_1740263

inputs
states_00
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 2
 matmul_1_readvariableop_resource:  
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ G
ReluReluadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/0
м,
“
while_body_1739876
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_11_matmul_readvariableop_resource_0: H
:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0: M
;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_11_matmul_readvariableop_resource: F
8while_simple_rnn_cell_11_biasadd_readvariableop_resource: K
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource:  ИҐ/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_11/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0®
.while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0≈
while/simple_rnn_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ¶
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ѕ
 while/simple_rnn_cell_11/BiasAddBiasAdd)while/simple_rnn_cell_11/MatMul:product:07while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ђ
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0ђ
!while/simple_rnn_cell_11/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ѓ
while/simple_rnn_cell_11/addAddV2)while/simple_rnn_cell_11/BiasAdd:output:0+while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ y
while/simple_rnn_cell_11/ReluRelu while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ‘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder+while/simple_rnn_cell_11/Relu:activations:0*
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
: И
while/Identity_4Identity+while/simple_rnn_cell_11/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ в

while/NoOpNoOp0^while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_11/MatMul/ReadVariableOp1^while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_11_matmul_readvariableop_resource9while_simple_rnn_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2b
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_11/MatMul/ReadVariableOp.while/simple_rnn_cell_11/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Ђ>
Љ
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1736231

inputsB
0simple_rnn_cell_9_matmul_readvariableop_resource: ?
1simple_rnn_cell_9_biasadd_readvariableop_resource:D
2simple_rnn_cell_9_matmul_1_readvariableop_resource:
identityИҐ(simple_rnn_cell_9/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_9/MatMul/ReadVariableOpҐ)simple_rnn_cell_9/MatMul_1/ReadVariableOpҐwhile;
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
value	B :s
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ D
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
valueB"€€€€    а
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
:€€€€€€€€€ *
shrink_axis_maskШ
'simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Я
simple_rnn_cell_9/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ц
(simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ђ
simple_rnn_cell_9/BiasAddBiasAdd"simple_rnn_cell_9/MatMul:product:00simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
)simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Щ
simple_rnn_cell_9/MatMul_1MatMulzeros:output:01simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
simple_rnn_cell_9/addAddV2"simple_rnn_cell_9/BiasAdd:output:0$simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€k
simple_rnn_cell_9/ReluRelusimple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_9_matmul_readvariableop_resource1simple_rnn_cell_9_biasadd_readvariableop_resource2simple_rnn_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1736164*
condR
while_cond_1736163*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
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
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ѕ
NoOpNoOp)^simple_rnn_cell_9/BiasAdd/ReadVariableOp(^simple_rnn_cell_9/MatMul/ReadVariableOp*^simple_rnn_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ : : : 2T
(simple_rnn_cell_9/BiasAdd/ReadVariableOp(simple_rnn_cell_9/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_9/MatMul/ReadVariableOp'simple_rnn_cell_9/MatMul/ReadVariableOp2V
)simple_rnn_cell_9/MatMul_1/ReadVariableOp)simple_rnn_cell_9/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
я
ѓ
while_cond_1739399
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1739399___redundant_placeholder05
1while_while_cond_1739399___redundant_placeholder15
1while_while_cond_1739399___redundant_placeholder25
1while_while_cond_1739399___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
я
ѓ
while_cond_1736395
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1736395___redundant_placeholder05
1while_while_cond_1736395___redundant_placeholder15
1while_while_cond_1736395___redundant_placeholder25
1while_while_cond_1736395___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
ј,
…
while_body_1738643
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0: L
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_8_matmul_readvariableop_resource: E
7while_simple_rnn_cell_8_biasadd_readvariableop_resource: J
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:  ИҐ.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_8/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0¶
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0√
while/simple_rnn_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ §
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Њ
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ™
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0™
 while/simple_rnn_cell_8/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ђ
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ w
while/simple_rnn_cell_8/ReluReluwhile/simple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ”
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_8/Relu:activations:0*
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
: З
while/Identity_4Identity*while/simple_rnn_cell_8/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ я

while/NoOpNoOp/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2`
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_8/MatMul/ReadVariableOp-while/simple_rnn_cell_8/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
Ї
∞
-sequential_19_simple_rnn_8_while_cond_1734335R
Nsequential_19_simple_rnn_8_while_sequential_19_simple_rnn_8_while_loop_counterX
Tsequential_19_simple_rnn_8_while_sequential_19_simple_rnn_8_while_maximum_iterations0
,sequential_19_simple_rnn_8_while_placeholder2
.sequential_19_simple_rnn_8_while_placeholder_12
.sequential_19_simple_rnn_8_while_placeholder_2T
Psequential_19_simple_rnn_8_while_less_sequential_19_simple_rnn_8_strided_slice_1k
gsequential_19_simple_rnn_8_while_sequential_19_simple_rnn_8_while_cond_1734335___redundant_placeholder0k
gsequential_19_simple_rnn_8_while_sequential_19_simple_rnn_8_while_cond_1734335___redundant_placeholder1k
gsequential_19_simple_rnn_8_while_sequential_19_simple_rnn_8_while_cond_1734335___redundant_placeholder2k
gsequential_19_simple_rnn_8_while_sequential_19_simple_rnn_8_while_cond_1734335___redundant_placeholder3-
)sequential_19_simple_rnn_8_while_identity
ќ
%sequential_19/simple_rnn_8/while/LessLess,sequential_19_simple_rnn_8_while_placeholderPsequential_19_simple_rnn_8_while_less_sequential_19_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: Б
)sequential_19/simple_rnn_8/while/IdentityIdentity)sequential_19/simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "_
)sequential_19_simple_rnn_8_while_identity2sequential_19/simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
ДG
Ю
.sequential_19_simple_rnn_10_while_body_1734550T
Psequential_19_simple_rnn_10_while_sequential_19_simple_rnn_10_while_loop_counterZ
Vsequential_19_simple_rnn_10_while_sequential_19_simple_rnn_10_while_maximum_iterations1
-sequential_19_simple_rnn_10_while_placeholder3
/sequential_19_simple_rnn_10_while_placeholder_13
/sequential_19_simple_rnn_10_while_placeholder_2S
Osequential_19_simple_rnn_10_while_sequential_19_simple_rnn_10_strided_slice_1_0Р
Лsequential_19_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0g
Usequential_19_simple_rnn_10_while_simple_rnn_cell_10_matmul_readvariableop_resource_0:d
Vsequential_19_simple_rnn_10_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:i
Wsequential_19_simple_rnn_10_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:.
*sequential_19_simple_rnn_10_while_identity0
,sequential_19_simple_rnn_10_while_identity_10
,sequential_19_simple_rnn_10_while_identity_20
,sequential_19_simple_rnn_10_while_identity_30
,sequential_19_simple_rnn_10_while_identity_4Q
Msequential_19_simple_rnn_10_while_sequential_19_simple_rnn_10_strided_slice_1О
Йsequential_19_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_10_tensorarrayunstack_tensorlistfromtensore
Ssequential_19_simple_rnn_10_while_simple_rnn_cell_10_matmul_readvariableop_resource:b
Tsequential_19_simple_rnn_10_while_simple_rnn_cell_10_biasadd_readvariableop_resource:g
Usequential_19_simple_rnn_10_while_simple_rnn_cell_10_matmul_1_readvariableop_resource:ИҐKsequential_19/simple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpҐJsequential_19/simple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOpҐLsequential_19/simple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp§
Ssequential_19/simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ≥
Esequential_19/simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЛsequential_19_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0-sequential_19_simple_rnn_10_while_placeholder\sequential_19/simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0а
Jsequential_19/simple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOpUsequential_19_simple_rnn_10_while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0Щ
;sequential_19/simple_rnn_10/while/simple_rnn_cell_10/MatMulMatMulLsequential_19/simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Rsequential_19/simple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ё
Ksequential_19/simple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOpVsequential_19_simple_rnn_10_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Х
<sequential_19/simple_rnn_10/while/simple_rnn_cell_10/BiasAddBiasAddEsequential_19/simple_rnn_10/while/simple_rnn_cell_10/MatMul:product:0Ssequential_19/simple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€д
Lsequential_19/simple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOpWsequential_19_simple_rnn_10_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0А
=sequential_19/simple_rnn_10/while/simple_rnn_cell_10/MatMul_1MatMul/sequential_19_simple_rnn_10_while_placeholder_2Tsequential_19/simple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Г
8sequential_19/simple_rnn_10/while/simple_rnn_cell_10/addAddV2Esequential_19/simple_rnn_10/while/simple_rnn_cell_10/BiasAdd:output:0Gsequential_19/simple_rnn_10/while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€±
9sequential_19/simple_rnn_10/while/simple_rnn_cell_10/ReluRelu<sequential_19/simple_rnn_10/while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ƒ
Fsequential_19/simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem/sequential_19_simple_rnn_10_while_placeholder_1-sequential_19_simple_rnn_10_while_placeholderGsequential_19/simple_rnn_10/while/simple_rnn_cell_10/Relu:activations:0*
_output_shapes
: *
element_dtype0:йи“i
'sequential_19/simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :∞
%sequential_19/simple_rnn_10/while/addAddV2-sequential_19_simple_rnn_10_while_placeholder0sequential_19/simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: k
)sequential_19/simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :„
'sequential_19/simple_rnn_10/while/add_1AddV2Psequential_19_simple_rnn_10_while_sequential_19_simple_rnn_10_while_loop_counter2sequential_19/simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: ≠
*sequential_19/simple_rnn_10/while/IdentityIdentity+sequential_19/simple_rnn_10/while/add_1:z:0'^sequential_19/simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Џ
,sequential_19/simple_rnn_10/while/Identity_1IdentityVsequential_19_simple_rnn_10_while_sequential_19_simple_rnn_10_while_maximum_iterations'^sequential_19/simple_rnn_10/while/NoOp*
T0*
_output_shapes
: ≠
,sequential_19/simple_rnn_10/while/Identity_2Identity)sequential_19/simple_rnn_10/while/add:z:0'^sequential_19/simple_rnn_10/while/NoOp*
T0*
_output_shapes
: Џ
,sequential_19/simple_rnn_10/while/Identity_3IdentityVsequential_19/simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^sequential_19/simple_rnn_10/while/NoOp*
T0*
_output_shapes
: №
,sequential_19/simple_rnn_10/while/Identity_4IdentityGsequential_19/simple_rnn_10/while/simple_rnn_cell_10/Relu:activations:0'^sequential_19/simple_rnn_10/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€“
&sequential_19/simple_rnn_10/while/NoOpNoOpL^sequential_19/simple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpK^sequential_19/simple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOpM^sequential_19/simple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "a
*sequential_19_simple_rnn_10_while_identity3sequential_19/simple_rnn_10/while/Identity:output:0"e
,sequential_19_simple_rnn_10_while_identity_15sequential_19/simple_rnn_10/while/Identity_1:output:0"e
,sequential_19_simple_rnn_10_while_identity_25sequential_19/simple_rnn_10/while/Identity_2:output:0"e
,sequential_19_simple_rnn_10_while_identity_35sequential_19/simple_rnn_10/while/Identity_3:output:0"e
,sequential_19_simple_rnn_10_while_identity_45sequential_19/simple_rnn_10/while/Identity_4:output:0"†
Msequential_19_simple_rnn_10_while_sequential_19_simple_rnn_10_strided_slice_1Osequential_19_simple_rnn_10_while_sequential_19_simple_rnn_10_strided_slice_1_0"Ѓ
Tsequential_19_simple_rnn_10_while_simple_rnn_cell_10_biasadd_readvariableop_resourceVsequential_19_simple_rnn_10_while_simple_rnn_cell_10_biasadd_readvariableop_resource_0"∞
Usequential_19_simple_rnn_10_while_simple_rnn_cell_10_matmul_1_readvariableop_resourceWsequential_19_simple_rnn_10_while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0"ђ
Ssequential_19_simple_rnn_10_while_simple_rnn_cell_10_matmul_readvariableop_resourceUsequential_19_simple_rnn_10_while_simple_rnn_cell_10_matmul_readvariableop_resource_0"Ъ
Йsequential_19_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorЛsequential_19_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_19_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2Ъ
Ksequential_19/simple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpKsequential_19/simple_rnn_10/while/simple_rnn_cell_10/BiasAdd/ReadVariableOp2Ш
Jsequential_19/simple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOpJsequential_19/simple_rnn_10/while/simple_rnn_cell_10/MatMul/ReadVariableOp2Ь
Lsequential_19/simple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOpLsequential_19/simple_rnn_10/while/simple_rnn_cell_10/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
і
ї
/__inference_simple_rnn_11_layer_call_fn_1739693
inputs_0
unknown: 
	unknown_0: 
	unknown_1:  
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1735751|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
Щ
м
O__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_1740466

inputs
states_00
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 2
 matmul_1_readvariableop_resource:  
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ G
ReluReluadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/0
И
ч
/__inference_sequential_19_layer_call_fn_1736509
simple_rnn_8_input
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8: 
	unknown_9: 

unknown_10:  

unknown_11: 

unknown_12:
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_19_layer_call_and_return_conditional_losses_1736478s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:€€€€€€€€€
,
_user_specified_namesimple_rnn_8_input
™4
§
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1735910

inputs,
simple_rnn_cell_11_1735835: (
simple_rnn_cell_11_1735837: ,
simple_rnn_cell_11_1735839:  
identityИҐ*simple_rnn_cell_11/StatefulPartitionedCallҐwhile;
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
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
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
valueB"€€€€   а
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
:€€€€€€€€€*
shrink_axis_maskр
*simple_rnn_cell_11/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_11_1735835simple_rnn_cell_11_1735837simple_rnn_cell_11_1735839*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_1735795n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Х
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_11_1735835simple_rnn_cell_11_1735837simple_rnn_cell_11_1735839*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1735847*
condR
while_cond_1735846*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
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
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ {
NoOpNoOp+^simple_rnn_cell_11/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2X
*simple_rnn_cell_11/StatefulPartitionedCall*simple_rnn_cell_11/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
м,
“
while_body_1736281
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:ИҐ/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_10/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0®
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0≈
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€¶
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ѕ
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0ђ
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ѓ
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€y
while/simple_rnn_cell_10/ReluRelu while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€‘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder+while/simple_rnn_cell_10/Relu:activations:0*
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
: И
while/Identity_4Identity+while/simple_rnn_cell_10/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€в

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
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2b
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
•=
Љ
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1737022

inputsB
0simple_rnn_cell_8_matmul_readvariableop_resource: ?
1simple_rnn_cell_8_biasadd_readvariableop_resource: D
2simple_rnn_cell_8_matmul_1_readvariableop_resource:  
identityИҐ(simple_rnn_cell_8/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_8/MatMul/ReadVariableOpҐ)simple_rnn_cell_8/MatMul_1/ReadVariableOpҐwhile;
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
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
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
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Я
simple_rnn_cell_8/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ц
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ђ
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ь
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0Щ
simple_rnn_cell_8/MatMul_1MatMulzeros:output:01simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ъ
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ k
simple_rnn_cell_8/ReluRelusimple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource2simple_rnn_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1736956*
condR
while_cond_1736955*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype0h
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
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ ѕ
NoOpNoOp)^simple_rnn_cell_8/BiasAdd/ReadVariableOp(^simple_rnn_cell_8/MatMul/ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2T
(simple_rnn_cell_8/BiasAdd/ReadVariableOp(simple_rnn_cell_8/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_8/MatMul/ReadVariableOp'simple_rnn_cell_8/MatMul/ReadVariableOp2V
)simple_rnn_cell_8/MatMul_1/ReadVariableOp)simple_rnn_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
м,
“
while_body_1739768
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_11_matmul_readvariableop_resource_0: H
:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0: M
;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_11_matmul_readvariableop_resource: F
8while_simple_rnn_cell_11_biasadd_readvariableop_resource: K
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource:  ИҐ/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_11/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0®
.while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0≈
while/simple_rnn_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ¶
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ѕ
 while/simple_rnn_cell_11/BiasAddBiasAdd)while/simple_rnn_cell_11/MatMul:product:07while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ђ
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0ђ
!while/simple_rnn_cell_11/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ѓ
while/simple_rnn_cell_11/addAddV2)while/simple_rnn_cell_11/BiasAdd:output:0+while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ y
while/simple_rnn_cell_11/ReluRelu while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ‘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder+while/simple_rnn_cell_11/Relu:activations:0*
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
: И
while/Identity_4Identity+while/simple_rnn_cell_11/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ в

while/NoOpNoOp0^while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_11/MatMul/ReadVariableOp1^while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_11_matmul_readvariableop_resource9while_simple_rnn_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2b
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_11/MatMul/ReadVariableOp.while/simple_rnn_cell_11/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
№-
…
while_body_1739016
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_9_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_9_matmul_readvariableop_resource: E
7while_simple_rnn_cell_9_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:ИҐ.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_9/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0¶
-while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0√
while/simple_rnn_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€§
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Њ
while/simple_rnn_cell_9/BiasAddBiasAdd(while/simple_rnn_cell_9/MatMul:product:06while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€™
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0™
 while/simple_rnn_cell_9/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
while/simple_rnn_cell_9/addAddV2(while/simple_rnn_cell_9/BiasAdd:output:0*while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€w
while/simple_rnn_cell_9/ReluReluwhile/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ы
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_9/Relu:activations:0*
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
: З
while/Identity_4Identity*while/simple_rnn_cell_9/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€я

while/NoOpNoOp/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_9_biasadd_readvariableop_resource9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_9_matmul_readvariableop_resource8while_simple_rnn_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2`
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_9/MatMul/ReadVariableOp-while/simple_rnn_cell_9/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
ј,
…
while_body_1738535
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0: L
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_8_matmul_readvariableop_resource: E
7while_simple_rnn_cell_8_biasadd_readvariableop_resource: J
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:  ИҐ.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_8/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0¶
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0√
while/simple_rnn_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ §
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Њ
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ™
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0™
 while/simple_rnn_cell_8/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ђ
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ w
while/simple_rnn_cell_8/ReluReluwhile/simple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ”
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_8/Relu:activations:0*
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
: З
while/Identity_4Identity*while/simple_rnn_cell_8/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ я

while/NoOpNoOp/^while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_8/MatMul/ReadVariableOp0^while/simple_rnn_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2`
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_8/MatMul/ReadVariableOp-while/simple_rnn_cell_8/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
я
ѓ
while_cond_1739291
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1739291___redundant_placeholder05
1while_while_cond_1739291___redundant_placeholder15
1while_while_cond_1739291___redundant_placeholder25
1while_while_cond_1739291___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
я
ѓ
while_cond_1735246
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1735246___redundant_placeholder05
1while_while_cond_1735246___redundant_placeholder15
1while_while_cond_1735246___redundant_placeholder25
1while_while_cond_1735246___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Д
•
P__inference_time_distributed_18_layer_call_and_return_conditional_losses_1740197

inputs9
'dense_21_matmul_readvariableop_resource: 6
(dense_21_biasadd_readvariableop_resource:
identityИҐdense_21/BiasAdd/ReadVariableOpҐdense_21/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
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
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ж
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Е
dense_21/MatMulMatMulReshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:И
	Reshape_1Reshapedense_21/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€Й
NoOpNoOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€ : : 2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Џ

є
 simple_rnn_11_while_cond_17377168
4simple_rnn_11_while_simple_rnn_11_while_loop_counter>
:simple_rnn_11_while_simple_rnn_11_while_maximum_iterations#
simple_rnn_11_while_placeholder%
!simple_rnn_11_while_placeholder_1%
!simple_rnn_11_while_placeholder_2:
6simple_rnn_11_while_less_simple_rnn_11_strided_slice_1Q
Msimple_rnn_11_while_simple_rnn_11_while_cond_1737716___redundant_placeholder0Q
Msimple_rnn_11_while_simple_rnn_11_while_cond_1737716___redundant_placeholder1Q
Msimple_rnn_11_while_simple_rnn_11_while_cond_1737716___redundant_placeholder2Q
Msimple_rnn_11_while_simple_rnn_11_while_cond_1737716___redundant_placeholder3 
simple_rnn_11_while_identity
Ъ
simple_rnn_11/while/LessLesssimple_rnn_11_while_placeholder6simple_rnn_11_while_less_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_11/while/IdentityIdentitysimple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_11_while_identity%simple_rnn_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
И
ч
/__inference_sequential_19_layer_call_fn_1737170
simple_rnn_8_input
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8: 
	unknown_9: 

unknown_10:  

unknown_11: 

unknown_12:
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_19_layer_call_and_return_conditional_losses_1737106s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:€€€€€€€€€
,
_user_specified_namesimple_rnn_8_input
Ъ4
Я
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1734856

inputs+
simple_rnn_cell_8_1734781: '
simple_rnn_cell_8_1734783: +
simple_rnn_cell_8_1734785:  
identityИҐ)simple_rnn_cell_8/StatefulPartitionedCallҐwhile;
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
:€€€€€€€€€ c
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
)simple_rnn_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_8_1734781simple_rnn_cell_8_1734783simple_rnn_cell_8_1734785*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_1734780n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_8_1734781simple_rnn_cell_8_1734783simple_rnn_cell_8_1734785*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1734793*
condR
while_cond_1734792*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
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
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ z
NoOpNoOp*^simple_rnn_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2V
)simple_rnn_cell_8/StatefulPartitionedCall)simple_rnn_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
оќ
у%
#__inference__traced_restore_1740812
file_prefixH
6assignvariableop_simple_rnn_8_simple_rnn_cell_8_kernel: T
Bassignvariableop_1_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel:  D
6assignvariableop_2_simple_rnn_8_simple_rnn_cell_8_bias: J
8assignvariableop_3_simple_rnn_9_simple_rnn_cell_9_kernel: T
Bassignvariableop_4_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel:D
6assignvariableop_5_simple_rnn_9_simple_rnn_cell_9_bias:L
:assignvariableop_6_simple_rnn_10_simple_rnn_cell_10_kernel:V
Dassignvariableop_7_simple_rnn_10_simple_rnn_cell_10_recurrent_kernel:F
8assignvariableop_8_simple_rnn_10_simple_rnn_cell_10_bias:L
:assignvariableop_9_simple_rnn_11_simple_rnn_cell_11_kernel: W
Eassignvariableop_10_simple_rnn_11_simple_rnn_cell_11_recurrent_kernel:  G
9assignvariableop_11_simple_rnn_11_simple_rnn_cell_11_bias: @
.assignvariableop_12_time_distributed_18_kernel: :
,assignvariableop_13_time_distributed_18_bias:'
assignvariableop_14_adam_iter:	 )
assignvariableop_15_adam_beta_1: )
assignvariableop_16_adam_beta_2: (
assignvariableop_17_adam_decay: 0
&assignvariableop_18_adam_learning_rate: #
assignvariableop_19_total: #
assignvariableop_20_count: R
@assignvariableop_21_adam_simple_rnn_8_simple_rnn_cell_8_kernel_m: \
Jassignvariableop_22_adam_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_m:  L
>assignvariableop_23_adam_simple_rnn_8_simple_rnn_cell_8_bias_m: R
@assignvariableop_24_adam_simple_rnn_9_simple_rnn_cell_9_kernel_m: \
Jassignvariableop_25_adam_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_m:L
>assignvariableop_26_adam_simple_rnn_9_simple_rnn_cell_9_bias_m:T
Bassignvariableop_27_adam_simple_rnn_10_simple_rnn_cell_10_kernel_m:^
Lassignvariableop_28_adam_simple_rnn_10_simple_rnn_cell_10_recurrent_kernel_m:N
@assignvariableop_29_adam_simple_rnn_10_simple_rnn_cell_10_bias_m:T
Bassignvariableop_30_adam_simple_rnn_11_simple_rnn_cell_11_kernel_m: ^
Lassignvariableop_31_adam_simple_rnn_11_simple_rnn_cell_11_recurrent_kernel_m:  N
@assignvariableop_32_adam_simple_rnn_11_simple_rnn_cell_11_bias_m: G
5assignvariableop_33_adam_time_distributed_18_kernel_m: A
3assignvariableop_34_adam_time_distributed_18_bias_m:R
@assignvariableop_35_adam_simple_rnn_8_simple_rnn_cell_8_kernel_v: \
Jassignvariableop_36_adam_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_v:  L
>assignvariableop_37_adam_simple_rnn_8_simple_rnn_cell_8_bias_v: R
@assignvariableop_38_adam_simple_rnn_9_simple_rnn_cell_9_kernel_v: \
Jassignvariableop_39_adam_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_v:L
>assignvariableop_40_adam_simple_rnn_9_simple_rnn_cell_9_bias_v:T
Bassignvariableop_41_adam_simple_rnn_10_simple_rnn_cell_10_kernel_v:^
Lassignvariableop_42_adam_simple_rnn_10_simple_rnn_cell_10_recurrent_kernel_v:N
@assignvariableop_43_adam_simple_rnn_10_simple_rnn_cell_10_bias_v:T
Bassignvariableop_44_adam_simple_rnn_11_simple_rnn_cell_11_kernel_v: ^
Lassignvariableop_45_adam_simple_rnn_11_simple_rnn_cell_11_recurrent_kernel_v:  N
@assignvariableop_46_adam_simple_rnn_11_simple_rnn_cell_11_bias_v: G
5assignvariableop_47_adam_time_distributed_18_kernel_v: A
3assignvariableop_48_adam_time_distributed_18_bias_v:
identity_50ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9К
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*∞
value¶B£2B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH‘
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ы
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ё
_output_shapesЋ
»::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOpAssignVariableOp6assignvariableop_simple_rnn_8_simple_rnn_cell_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_1AssignVariableOpBassignvariableop_1_simple_rnn_8_simple_rnn_cell_8_recurrent_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_2AssignVariableOp6assignvariableop_2_simple_rnn_8_simple_rnn_cell_8_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_3AssignVariableOp8assignvariableop_3_simple_rnn_9_simple_rnn_cell_9_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_4AssignVariableOpBassignvariableop_4_simple_rnn_9_simple_rnn_cell_9_recurrent_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_5AssignVariableOp6assignvariableop_5_simple_rnn_9_simple_rnn_cell_9_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_6AssignVariableOp:assignvariableop_6_simple_rnn_10_simple_rnn_cell_10_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_7AssignVariableOpDassignvariableop_7_simple_rnn_10_simple_rnn_cell_10_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_8AssignVariableOp8assignvariableop_8_simple_rnn_10_simple_rnn_cell_10_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_9AssignVariableOp:assignvariableop_9_simple_rnn_11_simple_rnn_cell_11_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_10AssignVariableOpEassignvariableop_10_simple_rnn_11_simple_rnn_cell_11_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_11AssignVariableOp9assignvariableop_11_simple_rnn_11_simple_rnn_cell_11_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_12AssignVariableOp.assignvariableop_12_time_distributed_18_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_13AssignVariableOp,assignvariableop_13_time_distributed_18_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_simple_rnn_8_simple_rnn_cell_8_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_22AssignVariableOpJassignvariableop_22_adam_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_simple_rnn_8_simple_rnn_cell_8_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_24AssignVariableOp@assignvariableop_24_adam_simple_rnn_9_simple_rnn_cell_9_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_25AssignVariableOpJassignvariableop_25_adam_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_26AssignVariableOp>assignvariableop_26_adam_simple_rnn_9_simple_rnn_cell_9_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_27AssignVariableOpBassignvariableop_27_adam_simple_rnn_10_simple_rnn_cell_10_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_28AssignVariableOpLassignvariableop_28_adam_simple_rnn_10_simple_rnn_cell_10_recurrent_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_simple_rnn_10_simple_rnn_cell_10_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_30AssignVariableOpBassignvariableop_30_adam_simple_rnn_11_simple_rnn_cell_11_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_31AssignVariableOpLassignvariableop_31_adam_simple_rnn_11_simple_rnn_cell_11_recurrent_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_32AssignVariableOp@assignvariableop_32_adam_simple_rnn_11_simple_rnn_cell_11_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_33AssignVariableOp5assignvariableop_33_adam_time_distributed_18_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_34AssignVariableOp3assignvariableop_34_adam_time_distributed_18_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_35AssignVariableOp@assignvariableop_35_adam_simple_rnn_8_simple_rnn_cell_8_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_36AssignVariableOpJassignvariableop_36_adam_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_37AssignVariableOp>assignvariableop_37_adam_simple_rnn_8_simple_rnn_cell_8_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_38AssignVariableOp@assignvariableop_38_adam_simple_rnn_9_simple_rnn_cell_9_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_39AssignVariableOpJassignvariableop_39_adam_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_40AssignVariableOp>assignvariableop_40_adam_simple_rnn_9_simple_rnn_cell_9_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_41AssignVariableOpBassignvariableop_41_adam_simple_rnn_10_simple_rnn_cell_10_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_42AssignVariableOpLassignvariableop_42_adam_simple_rnn_10_simple_rnn_cell_10_recurrent_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_43AssignVariableOp@assignvariableop_43_adam_simple_rnn_10_simple_rnn_cell_10_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_44AssignVariableOpBassignvariableop_44_adam_simple_rnn_11_simple_rnn_cell_11_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_45AssignVariableOpLassignvariableop_45_adam_simple_rnn_11_simple_rnn_cell_11_recurrent_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_46AssignVariableOp@assignvariableop_46_adam_simple_rnn_11_simple_rnn_cell_11_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_47AssignVariableOp5assignvariableop_47_adam_time_distributed_18_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_48AssignVariableOp3assignvariableop_48_adam_time_distributed_18_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Е	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_50IdentityIdentity_49:output:0^NoOp_1*
T0*
_output_shapes
: т
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_50Identity_50:output:0*w
_input_shapesf
d: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482(
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
я
ѓ
while_cond_1735554
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1735554___redundant_placeholder05
1while_while_cond_1735554___redundant_placeholder15
1while_while_cond_1735554___redundant_placeholder25
1while_while_cond_1735554___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
џ
N
2__inference_repeat_vector_18_layer_call_fn_1739198

inputs
identity≈
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_repeat_vector_18_layer_call_and_return_conditional_losses_1735332m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€€€€€€€€€€:X T
0
_output_shapes
:€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
я
ѓ
while_cond_1739507
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1739507___redundant_placeholder05
1while_while_cond_1739507___redundant_placeholder15
1while_while_cond_1739507___redundant_placeholder25
1while_while_cond_1739507___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
К
є
/__inference_simple_rnn_11_layer_call_fn_1739726

inputs
unknown: 
	unknown_0: 
	unknown_1:  
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1736630s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
У
к
O__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_1735795

inputs

states0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 2
 matmul_1_readvariableop_resource:  
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ G
ReluReluadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
г=
Њ
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1738385
inputs_0B
0simple_rnn_cell_8_matmul_readvariableop_resource: ?
1simple_rnn_cell_8_biasadd_readvariableop_resource: D
2simple_rnn_cell_8_matmul_1_readvariableop_resource:  
identityИҐ(simple_rnn_cell_8/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_8/MatMul/ReadVariableOpҐ)simple_rnn_cell_8/MatMul_1/ReadVariableOpҐwhile=
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
:€€€€€€€€€ c
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
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Я
simple_rnn_cell_8/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ц
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ђ
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ь
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0Щ
simple_rnn_cell_8/MatMul_1MatMulzeros:output:01simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ъ
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ k
simple_rnn_cell_8/ReluRelusimple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource2simple_rnn_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1738319*
condR
while_cond_1738318*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
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
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ ѕ
NoOpNoOp)^simple_rnn_cell_8/BiasAdd/ReadVariableOp(^simple_rnn_cell_8/MatMul/ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2T
(simple_rnn_cell_8/BiasAdd/ReadVariableOp(simple_rnn_cell_8/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_8/MatMul/ReadVariableOp'simple_rnn_cell_8/MatMul/ReadVariableOp2V
)simple_rnn_cell_8/MatMul_1/ReadVariableOp)simple_rnn_cell_8/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
х%
е
J__inference_sequential_19_layer_call_and_return_conditional_losses_1736478

inputs&
simple_rnn_8_1736115: "
simple_rnn_8_1736117: &
simple_rnn_8_1736119:  &
simple_rnn_9_1736232: "
simple_rnn_9_1736234:&
simple_rnn_9_1736236:'
simple_rnn_10_1736348:#
simple_rnn_10_1736350:'
simple_rnn_10_1736352:'
simple_rnn_11_1736463: #
simple_rnn_11_1736465: '
simple_rnn_11_1736467:  -
time_distributed_18_1736470: )
time_distributed_18_1736472:
identityИҐ%simple_rnn_10/StatefulPartitionedCallҐ%simple_rnn_11/StatefulPartitionedCallҐ$simple_rnn_8/StatefulPartitionedCallҐ$simple_rnn_9/StatefulPartitionedCallҐ+time_distributed_18/StatefulPartitionedCallЯ
$simple_rnn_8/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_8_1736115simple_rnn_8_1736117simple_rnn_8_1736119*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1736114¬
$simple_rnn_9/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_8/StatefulPartitionedCall:output:0simple_rnn_9_1736232simple_rnn_9_1736234simple_rnn_9_1736236*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1736231ф
 repeat_vector_18/PartitionedCallPartitionedCall-simple_rnn_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_repeat_vector_18_layer_call_and_return_conditional_losses_1735332«
%simple_rnn_10/StatefulPartitionedCallStatefulPartitionedCall)repeat_vector_18/PartitionedCall:output:0simple_rnn_10_1736348simple_rnn_10_1736350simple_rnn_10_1736352*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1736347ћ
%simple_rnn_11/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_10/StatefulPartitionedCall:output:0simple_rnn_11_1736463simple_rnn_11_1736465simple_rnn_11_1736467*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1736462Ћ
+time_distributed_18/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_11/StatefulPartitionedCall:output:0time_distributed_18_1736470time_distributed_18_1736472*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_time_distributed_18_layer_call_and_return_conditional_losses_1735954r
!time_distributed_18/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    і
time_distributed_18/ReshapeReshape.simple_rnn_11/StatefulPartitionedCall:output:0*time_distributed_18/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ З
IdentityIdentity4time_distributed_18/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Т
NoOpNoOp&^simple_rnn_10/StatefulPartitionedCall&^simple_rnn_11/StatefulPartitionedCall%^simple_rnn_8/StatefulPartitionedCall%^simple_rnn_9/StatefulPartitionedCall,^time_distributed_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 2N
%simple_rnn_10/StatefulPartitionedCall%simple_rnn_10/StatefulPartitionedCall2N
%simple_rnn_11/StatefulPartitionedCall%simple_rnn_11/StatefulPartitionedCall2L
$simple_rnn_8/StatefulPartitionedCall$simple_rnn_8/StatefulPartitionedCall2L
$simple_rnn_9/StatefulPartitionedCall$simple_rnn_9/StatefulPartitionedCall2Z
+time_distributed_18/StatefulPartitionedCall+time_distributed_18/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ш
Ї
.__inference_simple_rnn_9_layer_call_fn_1738731
inputs_0
unknown: 
	unknown_0:
	unknown_1:
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1735311o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
inputs/0
і
ї
/__inference_simple_rnn_10_layer_call_fn_1739228
inputs_0
unknown:
	unknown_0:
	unknown_1:
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1735618|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
я
ѓ
while_cond_1735846
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1735846___redundant_placeholder05
1while_while_cond_1735846___redundant_placeholder15
1while_while_cond_1735846___redundant_placeholder25
1while_while_cond_1735846___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Є
÷
P__inference_time_distributed_18_layer_call_and_return_conditional_losses_1735993

inputs"
dense_21_1735983: 
dense_21_1735985:
identityИҐ dense_21/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
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
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ э
 dense_21/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_21_1735983dense_21_1735985*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_1735943\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Х
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:Ш
	Reshape_1Reshape)dense_21/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€i
NoOpNoOp!^dense_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€ : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Ш
л
N__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_1740325

inputs
states_00
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€G
ReluReluadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€ :€€€€€€€€€: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states/0
ќ>
Њ
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1738973
inputs_0B
0simple_rnn_cell_9_matmul_readvariableop_resource: ?
1simple_rnn_cell_9_biasadd_readvariableop_resource:D
2simple_rnn_cell_9_matmul_1_readvariableop_resource:
identityИҐ(simple_rnn_cell_9/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_9/MatMul/ReadVariableOpҐ)simple_rnn_cell_9/MatMul_1/ReadVariableOpҐwhile=
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
value	B :s
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ D
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
valueB"€€€€    а
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
:€€€€€€€€€ *
shrink_axis_maskШ
'simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Я
simple_rnn_cell_9/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ц
(simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ђ
simple_rnn_cell_9/BiasAddBiasAdd"simple_rnn_cell_9/MatMul:product:00simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
)simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_9_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Щ
simple_rnn_cell_9/MatMul_1MatMulzeros:output:01simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
simple_rnn_cell_9/addAddV2"simple_rnn_cell_9/BiasAdd:output:0$simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€k
simple_rnn_cell_9/ReluRelusimple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_9_matmul_readvariableop_resource1simple_rnn_cell_9_biasadd_readvariableop_resource2simple_rnn_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1738906*
condR
while_cond_1738905*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
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
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ѕ
NoOpNoOp)^simple_rnn_cell_9/BiasAdd/ReadVariableOp(^simple_rnn_cell_9/MatMul/ReadVariableOp*^simple_rnn_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 2T
(simple_rnn_cell_9/BiasAdd/ReadVariableOp(simple_rnn_cell_9/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_9/MatMul/ReadVariableOp'simple_rnn_cell_9/MatMul/ReadVariableOp2V
)simple_rnn_cell_9/MatMul_1/ReadVariableOp)simple_rnn_cell_9/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
inputs/0
 =
√
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1736462

inputsC
1simple_rnn_cell_11_matmul_readvariableop_resource: @
2simple_rnn_cell_11_biasadd_readvariableop_resource: E
3simple_rnn_cell_11_matmul_1_readvariableop_resource:  
identityИҐ)simple_rnn_cell_11/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_11/MatMul/ReadVariableOpҐ*simple_rnn_cell_11/MatMul_1/ReadVariableOpҐwhile;
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
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
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
valueB"€€€€   а
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
:€€€€€€€€€*
shrink_axis_maskЪ
(simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

: *
dtype0°
simple_rnn_cell_11/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ш
)simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ѓ
simple_rnn_cell_11/BiasAddBiasAdd#simple_rnn_cell_11/MatMul:product:01simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ю
*simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0Ы
simple_rnn_cell_11/MatMul_1MatMulzeros:output:02simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Э
simple_rnn_cell_11/addAddV2#simple_rnn_cell_11/BiasAdd:output:0%simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ m
simple_rnn_cell_11/ReluRelusimple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_11_matmul_readvariableop_resource2simple_rnn_cell_11_biasadd_readvariableop_resource3simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1736396*
condR
while_cond_1736395*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype0h
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
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ “
NoOpNoOp*^simple_rnn_cell_11/BiasAdd/ReadVariableOp)^simple_rnn_cell_11/MatMul/ReadVariableOp+^simple_rnn_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2V
)simple_rnn_cell_11/BiasAdd/ReadVariableOp)simple_rnn_cell_11/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_11/MatMul/ReadVariableOp(simple_rnn_cell_11/MatMul/ReadVariableOp2X
*simple_rnn_cell_11/MatMul_1/ReadVariableOp*simple_rnn_cell_11/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
О
Ґ
5__inference_time_distributed_18_layer_call_fn_1740176

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_time_distributed_18_layer_call_and_return_conditional_losses_1735993|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
А
Є
.__inference_simple_rnn_9_layer_call_fn_1738753

inputs
unknown: 
	unknown_0:
	unknown_1:
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1736892o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
я
ѓ
while_cond_1735085
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1735085___redundant_placeholder05
1while_while_cond_1735085___redundant_placeholder15
1while_while_cond_1735085___redundant_placeholder25
1while_while_cond_1735085___redundant_placeholder3
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
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Е5
Я
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1735150

inputs+
simple_rnn_cell_9_1735073: '
simple_rnn_cell_9_1735075:+
simple_rnn_cell_9_1735077:
identityИҐ)simple_rnn_cell_9/StatefulPartitionedCallҐwhile;
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
value	B :s
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ D
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
valueB"€€€€    а
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
:€€€€€€€€€ *
shrink_axis_maskл
)simple_rnn_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_9_1735073simple_rnn_cell_9_1735075simple_rnn_cell_9_1735077*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_1735072n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_9_1735073simple_rnn_cell_9_1735075simple_rnn_cell_9_1735077*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1735086*
condR
while_cond_1735085*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
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
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€z
NoOpNoOp*^simple_rnn_cell_9/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 2V
)simple_rnn_cell_9/StatefulPartitionedCall)simple_rnn_cell_9/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
И>
≈
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1739834
inputs_0C
1simple_rnn_cell_11_matmul_readvariableop_resource: @
2simple_rnn_cell_11_biasadd_readvariableop_resource: E
3simple_rnn_cell_11_matmul_1_readvariableop_resource:  
identityИҐ)simple_rnn_cell_11/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_11/MatMul/ReadVariableOpҐ*simple_rnn_cell_11/MatMul_1/ReadVariableOpҐwhile=
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
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
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
valueB"€€€€   а
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
:€€€€€€€€€*
shrink_axis_maskЪ
(simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

: *
dtype0°
simple_rnn_cell_11/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ш
)simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ѓ
simple_rnn_cell_11/BiasAddBiasAdd#simple_rnn_cell_11/MatMul:product:01simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ю
*simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0Ы
simple_rnn_cell_11/MatMul_1MatMulzeros:output:02simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Э
simple_rnn_cell_11/addAddV2#simple_rnn_cell_11/BiasAdd:output:0%simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ m
simple_rnn_cell_11/ReluRelusimple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_11_matmul_readvariableop_resource2simple_rnn_cell_11_biasadd_readvariableop_resource3simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1739768*
condR
while_cond_1739767*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
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
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ “
NoOpNoOp*^simple_rnn_cell_11/BiasAdd/ReadVariableOp)^simple_rnn_cell_11/MatMul/ReadVariableOp+^simple_rnn_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2V
)simple_rnn_cell_11/BiasAdd/ReadVariableOp)simple_rnn_cell_11/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_11/MatMul/ReadVariableOp(simple_rnn_cell_11/MatMul/ReadVariableOp2X
*simple_rnn_cell_11/MatMul_1/ReadVariableOp*simple_rnn_cell_11/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0
•=
Љ
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1738709

inputsB
0simple_rnn_cell_8_matmul_readvariableop_resource: ?
1simple_rnn_cell_8_biasadd_readvariableop_resource: D
2simple_rnn_cell_8_matmul_1_readvariableop_resource:  
identityИҐ(simple_rnn_cell_8/BiasAdd/ReadVariableOpҐ'simple_rnn_cell_8/MatMul/ReadVariableOpҐ)simple_rnn_cell_8/MatMul_1/ReadVariableOpҐwhile;
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
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
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
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Я
simple_rnn_cell_8/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ц
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ђ
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ь
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0Щ
simple_rnn_cell_8/MatMul_1MatMulzeros:output:01simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ъ
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ k
simple_rnn_cell_8/ReluRelusimple_rnn_cell_8/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource2simple_rnn_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1738643*
condR
while_cond_1738642*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype0h
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
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ ѕ
NoOpNoOp)^simple_rnn_cell_8/BiasAdd/ReadVariableOp(^simple_rnn_cell_8/MatMul/ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2T
(simple_rnn_cell_8/BiasAdd/ReadVariableOp(simple_rnn_cell_8/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_8/MatMul/ReadVariableOp'simple_rnn_cell_8/MatMul/ReadVariableOp2V
)simple_rnn_cell_8/MatMul_1/ReadVariableOp)simple_rnn_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
О
Ґ
5__inference_time_distributed_18_layer_call_fn_1740167

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_time_distributed_18_layer_call_and_return_conditional_losses_1735954|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Џ

є
 simple_rnn_10_while_cond_17376128
4simple_rnn_10_while_simple_rnn_10_while_loop_counter>
:simple_rnn_10_while_simple_rnn_10_while_maximum_iterations#
simple_rnn_10_while_placeholder%
!simple_rnn_10_while_placeholder_1%
!simple_rnn_10_while_placeholder_2:
6simple_rnn_10_while_less_simple_rnn_10_strided_slice_1Q
Msimple_rnn_10_while_simple_rnn_10_while_cond_1737612___redundant_placeholder0Q
Msimple_rnn_10_while_simple_rnn_10_while_cond_1737612___redundant_placeholder1Q
Msimple_rnn_10_while_simple_rnn_10_while_cond_1737612___redundant_placeholder2Q
Msimple_rnn_10_while_simple_rnn_10_while_cond_1737612___redundant_placeholder3 
simple_rnn_10_while_identity
Ъ
simple_rnn_10/while/LessLesssimple_rnn_10_while_placeholder6simple_rnn_10_while_less_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_10/while/IdentityIdentitysimple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_10_while_identity%simple_rnn_10/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
ї

џ
3__inference_simple_rnn_cell_8_layer_call_fn_1740232

inputs
states_0
unknown: 
	unknown_0: 
	unknown_1:  
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
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_1734780o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:€€€€€€€€€:€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
states/0
м,
“
while_body_1736694
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:ИҐ/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_10/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0®
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0≈
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€¶
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ѕ
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0ђ
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ѓ
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€y
while/simple_rnn_cell_10/ReluRelu while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€‘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder+while/simple_rnn_cell_10/Relu:activations:0*
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
: И
while/Identity_4Identity+while/simple_rnn_cell_10/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€в

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
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2b
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
™4
§
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1735751

inputs,
simple_rnn_cell_11_1735676: (
simple_rnn_cell_11_1735678: ,
simple_rnn_cell_11_1735680:  
identityИҐ*simple_rnn_cell_11/StatefulPartitionedCallҐwhile;
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
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
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
valueB"€€€€   а
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
:€€€€€€€€€*
shrink_axis_maskр
*simple_rnn_cell_11/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_11_1735676simple_rnn_cell_11_1735678simple_rnn_cell_11_1735680*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€ :€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_1735675n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Х
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_11_1735676simple_rnn_cell_11_1735678simple_rnn_cell_11_1735680*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1735688*
condR
while_cond_1735687*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
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
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ {
NoOpNoOp+^simple_rnn_cell_11/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2X
*simple_rnn_cell_11/StatefulPartitionedCall*simple_rnn_cell_11/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
я
ѓ
while_cond_1734792
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1734792___redundant_placeholder05
1while_while_cond_1734792___redundant_placeholder15
1while_while_cond_1734792___redundant_placeholder25
1while_while_cond_1734792___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
њ

¶
simple_rnn_8_while_cond_17378366
2simple_rnn_8_while_simple_rnn_8_while_loop_counter<
8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations"
simple_rnn_8_while_placeholder$
 simple_rnn_8_while_placeholder_1$
 simple_rnn_8_while_placeholder_28
4simple_rnn_8_while_less_simple_rnn_8_strided_slice_1O
Ksimple_rnn_8_while_simple_rnn_8_while_cond_1737836___redundant_placeholder0O
Ksimple_rnn_8_while_simple_rnn_8_while_cond_1737836___redundant_placeholder1O
Ksimple_rnn_8_while_simple_rnn_8_while_cond_1737836___redundant_placeholder2O
Ksimple_rnn_8_while_simple_rnn_8_while_cond_1737836___redundant_placeholder3
simple_rnn_8_while_identity
Ц
simple_rnn_8/while/LessLesssimple_rnn_8_while_placeholder4simple_rnn_8_while_less_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_8/while/IdentityIdentitysimple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_8_while_identity$simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
 =
√
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1736630

inputsC
1simple_rnn_cell_11_matmul_readvariableop_resource: @
2simple_rnn_cell_11_biasadd_readvariableop_resource: E
3simple_rnn_cell_11_matmul_1_readvariableop_resource:  
identityИҐ)simple_rnn_cell_11/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_11/MatMul/ReadVariableOpҐ*simple_rnn_cell_11/MatMul_1/ReadVariableOpҐwhile;
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
:€€€€€€€€€ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
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
valueB"€€€€   а
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
:€€€€€€€€€*
shrink_axis_maskЪ
(simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_11_matmul_readvariableop_resource*
_output_shapes

: *
dtype0°
simple_rnn_cell_11/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ш
)simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ѓ
simple_rnn_cell_11/BiasAddBiasAdd#simple_rnn_cell_11/MatMul:product:01simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ю
*simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0Ы
simple_rnn_cell_11/MatMul_1MatMulzeros:output:02simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Э
simple_rnn_cell_11/addAddV2#simple_rnn_cell_11/BiasAdd:output:0%simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ m
simple_rnn_cell_11/ReluRelusimple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_11_matmul_readvariableop_resource2simple_rnn_cell_11_biasadd_readvariableop_resource3simple_rnn_cell_11_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1736564*
condR
while_cond_1736563*8
output_shapes'
%: : : : :€€€€€€€€€ : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype0h
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
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ “
NoOpNoOp*^simple_rnn_cell_11/BiasAdd/ReadVariableOp)^simple_rnn_cell_11/MatMul/ReadVariableOp+^simple_rnn_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2V
)simple_rnn_cell_11/BiasAdd/ReadVariableOp)simple_rnn_cell_11/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_11/MatMul/ReadVariableOp(simple_rnn_cell_11/MatMul/ReadVariableOp2X
*simple_rnn_cell_11/MatMul_1/ReadVariableOp*simple_rnn_cell_11/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
њ

¶
simple_rnn_8_while_cond_17373986
2simple_rnn_8_while_simple_rnn_8_while_loop_counter<
8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations"
simple_rnn_8_while_placeholder$
 simple_rnn_8_while_placeholder_1$
 simple_rnn_8_while_placeholder_28
4simple_rnn_8_while_less_simple_rnn_8_strided_slice_1O
Ksimple_rnn_8_while_simple_rnn_8_while_cond_1737398___redundant_placeholder0O
Ksimple_rnn_8_while_simple_rnn_8_while_cond_1737398___redundant_placeholder1O
Ksimple_rnn_8_while_simple_rnn_8_while_cond_1737398___redundant_placeholder2O
Ksimple_rnn_8_while_simple_rnn_8_while_cond_1737398___redundant_placeholder3
simple_rnn_8_while_identity
Ц
simple_rnn_8/while/LessLesssimple_rnn_8_while_placeholder4simple_rnn_8_while_less_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_8/while/IdentityIdentitysimple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_8_while_identity$simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
Ї
∞
-sequential_19_simple_rnn_9_while_cond_1734440R
Nsequential_19_simple_rnn_9_while_sequential_19_simple_rnn_9_while_loop_counterX
Tsequential_19_simple_rnn_9_while_sequential_19_simple_rnn_9_while_maximum_iterations0
,sequential_19_simple_rnn_9_while_placeholder2
.sequential_19_simple_rnn_9_while_placeholder_12
.sequential_19_simple_rnn_9_while_placeholder_2T
Psequential_19_simple_rnn_9_while_less_sequential_19_simple_rnn_9_strided_slice_1k
gsequential_19_simple_rnn_9_while_sequential_19_simple_rnn_9_while_cond_1734440___redundant_placeholder0k
gsequential_19_simple_rnn_9_while_sequential_19_simple_rnn_9_while_cond_1734440___redundant_placeholder1k
gsequential_19_simple_rnn_9_while_sequential_19_simple_rnn_9_while_cond_1734440___redundant_placeholder2k
gsequential_19_simple_rnn_9_while_sequential_19_simple_rnn_9_while_cond_1734440___redundant_placeholder3-
)sequential_19_simple_rnn_9_while_identity
ќ
%sequential_19/simple_rnn_9/while/LessLess,sequential_19_simple_rnn_9_while_placeholderPsequential_19_simple_rnn_9_while_less_sequential_19_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: Б
)sequential_19/simple_rnn_9/while/IdentityIdentity)sequential_19/simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "_
)sequential_19_simple_rnn_9_while_identity2sequential_19/simple_rnn_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :€€€€€€€€€: ::::: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
№-
…
while_body_1738796
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_9_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_9_matmul_readvariableop_resource: E
7while_simple_rnn_cell_9_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:ИҐ.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpҐ-while/simple_rnn_cell_9/MatMul/ReadVariableOpҐ/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0¶
-while/simple_rnn_cell_9/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_9_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0√
while/simple_rnn_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€§
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Њ
while/simple_rnn_cell_9/BiasAddBiasAdd(while/simple_rnn_cell_9/MatMul:product:06while/simple_rnn_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€™
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0™
 while/simple_rnn_cell_9/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
while/simple_rnn_cell_9/addAddV2(while/simple_rnn_cell_9/BiasAdd:output:0*while/simple_rnn_cell_9/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€w
while/simple_rnn_cell_9/ReluReluwhile/simple_rnn_cell_9/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ы
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_9/Relu:activations:0*
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
: З
while/Identity_4Identity*while/simple_rnn_cell_9/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€я

while/NoOpNoOp/^while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_9/MatMul/ReadVariableOp0^while/simple_rnn_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_9_biasadd_readvariableop_resource9while_simple_rnn_cell_9_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_9_matmul_1_readvariableop_resource:while_simple_rnn_cell_9_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_9_matmul_readvariableop_resource8while_simple_rnn_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2`
.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp.while/simple_rnn_cell_9/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_9/MatMul/ReadVariableOp-while/simple_rnn_cell_9/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp/while/simple_rnn_cell_9/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
ыm
Х
 __inference__traced_save_1740655
file_prefixD
@savev2_simple_rnn_8_simple_rnn_cell_8_kernel_read_readvariableopN
Jsavev2_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_8_simple_rnn_cell_8_bias_read_readvariableopD
@savev2_simple_rnn_9_simple_rnn_cell_9_kernel_read_readvariableopN
Jsavev2_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_9_simple_rnn_cell_9_bias_read_readvariableopF
Bsavev2_simple_rnn_10_simple_rnn_cell_10_kernel_read_readvariableopP
Lsavev2_simple_rnn_10_simple_rnn_cell_10_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_10_simple_rnn_cell_10_bias_read_readvariableopF
Bsavev2_simple_rnn_11_simple_rnn_cell_11_kernel_read_readvariableopP
Lsavev2_simple_rnn_11_simple_rnn_cell_11_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_11_simple_rnn_cell_11_bias_read_readvariableop9
5savev2_time_distributed_18_kernel_read_readvariableop7
3savev2_time_distributed_18_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopK
Gsavev2_adam_simple_rnn_8_simple_rnn_cell_8_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_8_simple_rnn_cell_8_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_9_simple_rnn_cell_9_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_9_simple_rnn_cell_9_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_10_simple_rnn_cell_10_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_10_simple_rnn_cell_10_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_10_simple_rnn_cell_10_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_11_simple_rnn_cell_11_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_11_simple_rnn_cell_11_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_11_simple_rnn_cell_11_bias_m_read_readvariableop@
<savev2_adam_time_distributed_18_kernel_m_read_readvariableop>
:savev2_adam_time_distributed_18_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_8_simple_rnn_cell_8_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_8_simple_rnn_cell_8_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_9_simple_rnn_cell_9_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_9_simple_rnn_cell_9_bias_v_read_readvariableopM
Isavev2_adam_simple_rnn_10_simple_rnn_cell_10_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_10_simple_rnn_cell_10_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_10_simple_rnn_cell_10_bias_v_read_readvariableopM
Isavev2_adam_simple_rnn_11_simple_rnn_cell_11_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_11_simple_rnn_cell_11_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_11_simple_rnn_cell_11_bias_v_read_readvariableop@
<savev2_adam_time_distributed_18_kernel_v_read_readvariableop>
:savev2_adam_time_distributed_18_bias_v_read_readvariableop
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
: З
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*∞
value¶B£2B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH—
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ≈
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0@savev2_simple_rnn_8_simple_rnn_cell_8_kernel_read_readvariableopJsavev2_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_read_readvariableop>savev2_simple_rnn_8_simple_rnn_cell_8_bias_read_readvariableop@savev2_simple_rnn_9_simple_rnn_cell_9_kernel_read_readvariableopJsavev2_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_read_readvariableop>savev2_simple_rnn_9_simple_rnn_cell_9_bias_read_readvariableopBsavev2_simple_rnn_10_simple_rnn_cell_10_kernel_read_readvariableopLsavev2_simple_rnn_10_simple_rnn_cell_10_recurrent_kernel_read_readvariableop@savev2_simple_rnn_10_simple_rnn_cell_10_bias_read_readvariableopBsavev2_simple_rnn_11_simple_rnn_cell_11_kernel_read_readvariableopLsavev2_simple_rnn_11_simple_rnn_cell_11_recurrent_kernel_read_readvariableop@savev2_simple_rnn_11_simple_rnn_cell_11_bias_read_readvariableop5savev2_time_distributed_18_kernel_read_readvariableop3savev2_time_distributed_18_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopGsavev2_adam_simple_rnn_8_simple_rnn_cell_8_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_8_simple_rnn_cell_8_bias_m_read_readvariableopGsavev2_adam_simple_rnn_9_simple_rnn_cell_9_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_9_simple_rnn_cell_9_bias_m_read_readvariableopIsavev2_adam_simple_rnn_10_simple_rnn_cell_10_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_10_simple_rnn_cell_10_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_10_simple_rnn_cell_10_bias_m_read_readvariableopIsavev2_adam_simple_rnn_11_simple_rnn_cell_11_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_11_simple_rnn_cell_11_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_11_simple_rnn_cell_11_bias_m_read_readvariableop<savev2_adam_time_distributed_18_kernel_m_read_readvariableop:savev2_adam_time_distributed_18_bias_m_read_readvariableopGsavev2_adam_simple_rnn_8_simple_rnn_cell_8_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_8_simple_rnn_cell_8_bias_v_read_readvariableopGsavev2_adam_simple_rnn_9_simple_rnn_cell_9_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_9_simple_rnn_cell_9_bias_v_read_readvariableopIsavev2_adam_simple_rnn_10_simple_rnn_cell_10_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_10_simple_rnn_cell_10_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_10_simple_rnn_cell_10_bias_v_read_readvariableopIsavev2_adam_simple_rnn_11_simple_rnn_cell_11_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_11_simple_rnn_cell_11_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_11_simple_rnn_cell_11_bias_v_read_readvariableop<savev2_adam_time_distributed_18_kernel_v_read_readvariableop:savev2_adam_time_distributed_18_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	Р
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

identity_1Identity_1:output:0*П
_input_shapesэ
ъ: : :  : : :::::: :  : : :: : : : : : : : :  : : :::::: :  : : :: :  : : :::::: :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 	

_output_shapes
::$
 

_output_shapes

: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::
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
: :$ 

_output_shapes

: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: :$  

_output_shapes

:  : !

_output_shapes
: :$" 

_output_shapes

: : #

_output_shapes
::$$ 

_output_shapes

: :$% 

_output_shapes

:  : &

_output_shapes
: :$' 

_output_shapes

: :$( 

_output_shapes

:: )

_output_shapes
::$* 

_output_shapes

::$+ 

_output_shapes

:: ,

_output_shapes
::$- 

_output_shapes

: :$. 

_output_shapes

:  : /

_output_shapes
: :$0 

_output_shapes

: : 1

_output_shapes
::2

_output_shapes
: 
£"
Ў
while_body_1735247
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
!while_simple_rnn_cell_9_1735269_0: /
!while_simple_rnn_cell_9_1735271_0:3
!while_simple_rnn_cell_9_1735273_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
while_simple_rnn_cell_9_1735269: -
while_simple_rnn_cell_9_1735271:1
while_simple_rnn_cell_9_1735273:ИҐ/while/simple_rnn_cell_9/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype0¶
/while/simple_rnn_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_9_1735269_0!while_simple_rnn_cell_9_1735271_0!while_simple_rnn_cell_9_1735273_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_1735194r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Й
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:08while/simple_rnn_cell_9/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity8while/simple_rnn_cell_9/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€~

while/NoOpNoOp0^while/simple_rnn_cell_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_9_1735269!while_simple_rnn_cell_9_1735269_0"D
while_simple_rnn_cell_9_1735271!while_simple_rnn_cell_9_1735271_0"D
while_simple_rnn_cell_9_1735273!while_simple_rnn_cell_9_1735273_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2b
/while/simple_rnn_cell_9/StatefulPartitionedCall/while/simple_rnn_cell_9/StatefulPartitionedCall: 
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
 =
√
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1736760

inputsC
1simple_rnn_cell_10_matmul_readvariableop_resource:@
2simple_rnn_cell_10_biasadd_readvariableop_resource:E
3simple_rnn_cell_10_matmul_1_readvariableop_resource:
identityИҐ)simple_rnn_cell_10/BiasAdd/ReadVariableOpҐ(simple_rnn_cell_10/MatMul/ReadVariableOpҐ*simple_rnn_cell_10/MatMul_1/ReadVariableOpҐwhile;
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
value	B :s
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
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
valueB"€€€€   а
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
:€€€€€€€€€*
shrink_axis_maskЪ
(simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype0°
simple_rnn_cell_10/MatMulMatMulstrided_slice_2:output:00simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
simple_rnn_cell_10/BiasAddBiasAdd#simple_rnn_cell_10/MatMul:product:01simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
*simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_10_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Ы
simple_rnn_cell_10/MatMul_1MatMulzeros:output:02simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
simple_rnn_cell_10/addAddV2#simple_rnn_cell_10/BiasAdd:output:0%simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€m
simple_rnn_cell_10/ReluRelusimple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Ё
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_10_matmul_readvariableop_resource2simple_rnn_cell_10_biasadd_readvariableop_resource3simple_rnn_cell_10_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1736694*
condR
while_cond_1736693*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0h
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
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€“
NoOpNoOp*^simple_rnn_cell_10/BiasAdd/ReadVariableOp)^simple_rnn_cell_10/MatMul/ReadVariableOp+^simple_rnn_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2V
)simple_rnn_cell_10/BiasAdd/ReadVariableOp)simple_rnn_cell_10/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_10/MatMul/ReadVariableOp(simple_rnn_cell_10/MatMul/ReadVariableOp2X
*simple_rnn_cell_10/MatMul_1/ReadVariableOp*simple_rnn_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
м,
“
while_body_1736396
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_11_matmul_readvariableop_resource_0: H
:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0: M
;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_11_matmul_readvariableop_resource: F
8while_simple_rnn_cell_11_biasadd_readvariableop_resource: K
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource:  ИҐ/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_11/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0®
.while/simple_rnn_cell_11/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_11_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0≈
while/simple_rnn_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ¶
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ѕ
 while/simple_rnn_cell_11/BiasAddBiasAdd)while/simple_rnn_cell_11/MatMul:product:07while/simple_rnn_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ђ
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0ђ
!while/simple_rnn_cell_11/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ѓ
while/simple_rnn_cell_11/addAddV2)while/simple_rnn_cell_11/BiasAdd:output:0+while/simple_rnn_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€ y
while/simple_rnn_cell_11/ReluRelu while/simple_rnn_cell_11/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€ ‘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder+while/simple_rnn_cell_11/Relu:activations:0*
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
: И
while/Identity_4Identity+while/simple_rnn_cell_11/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ в

while/NoOpNoOp0^while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_11/MatMul/ReadVariableOp1^while/simple_rnn_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_11_biasadd_readvariableop_resource:while_simple_rnn_cell_11_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_11_matmul_1_readvariableop_resource;while_simple_rnn_cell_11_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_11_matmul_readvariableop_resource9while_simple_rnn_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€ : : : : : 2b
/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp/while/simple_rnn_cell_11/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_11/MatMul/ReadVariableOp.while/simple_rnn_cell_11/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp0while/simple_rnn_cell_11/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: 
м,
“
while_body_1739400
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
9while_simple_rnn_cell_10_matmul_readvariableop_resource_0:H
:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0:M
;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
7while_simple_rnn_cell_10_matmul_readvariableop_resource:F
8while_simple_rnn_cell_10_biasadd_readvariableop_resource:K
9while_simple_rnn_cell_10_matmul_1_readvariableop_resource:ИҐ/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpҐ.while/simple_rnn_cell_10/MatMul/ReadVariableOpҐ0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0®
.while/simple_rnn_cell_10/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_10_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0≈
while/simple_rnn_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€¶
/while/simple_rnn_cell_10/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_10_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Ѕ
 while/simple_rnn_cell_10/BiasAddBiasAdd)while/simple_rnn_cell_10/MatMul:product:07while/simple_rnn_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
0while/simple_rnn_cell_10/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0ђ
!while/simple_rnn_cell_10/MatMul_1MatMulwhile_placeholder_28while/simple_rnn_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ѓ
while/simple_rnn_cell_10/addAddV2)while/simple_rnn_cell_10/BiasAdd:output:0+while/simple_rnn_cell_10/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€y
while/simple_rnn_cell_10/ReluRelu while/simple_rnn_cell_10/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€‘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder+while/simple_rnn_cell_10/Relu:activations:0*
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
: И
while/Identity_4Identity+while/simple_rnn_cell_10/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€в

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
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :€€€€€€€€€: : : : : 2b
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
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
я
ѓ
while_cond_1740091
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1740091___redundant_placeholder05
1while_while_cond_1740091___redundant_placeholder15
1while_while_cond_1740091___redundant_placeholder25
1while_while_cond_1740091___redundant_placeholder3
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
-: : : : :€€€€€€€€€ : ::::: 
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
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
:
™4
§
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1735618

inputs,
simple_rnn_cell_10_1735543:(
simple_rnn_cell_10_1735545:,
simple_rnn_cell_10_1735547:
identityИҐ*simple_rnn_cell_10/StatefulPartitionedCallҐwhile;
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
value	B :s
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
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
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
valueB"€€€€   а
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
:€€€€€€€€€*
shrink_axis_maskр
*simple_rnn_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_10_1735543simple_rnn_cell_10_1735545simple_rnn_cell_10_1735547*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_1735503n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Х
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_10_1735543simple_rnn_cell_10_1735545simple_rnn_cell_10_1735547*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1735555*
condR
while_cond_1735554*8
output_shapes'
%: : : : :€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
element_dtype0h
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
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€{
NoOpNoOp+^simple_rnn_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2X
*simple_rnn_cell_10/StatefulPartitionedCall*simple_rnn_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*‘
serving_defaultј
U
simple_rnn_8_input?
$serving_default_simple_rnn_8_input:0€€€€€€€€€K
time_distributed_184
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:ії
ґ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
√
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec"
_tf_keras_rnn_layer
√
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec"
_tf_keras_rnn_layer
•
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
√
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
√
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4cell
5
state_spec"
_tf_keras_rnn_layer
∞
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
	<layer"
_tf_keras_layer
Ж
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
I12
J13"
trackable_list_wrapper
Ж
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
I12
J13"
trackable_list_wrapper
 "
trackable_list_wrapper
 
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
с
Ptrace_0
Qtrace_1
Rtrace_2
Strace_32Ж
/__inference_sequential_19_layer_call_fn_1736509
/__inference_sequential_19_layer_call_fn_1737324
/__inference_sequential_19_layer_call_fn_1737357
/__inference_sequential_19_layer_call_fn_1737170њ
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
 zPtrace_0zQtrace_1zRtrace_2zStrace_3
Ё
Ttrace_0
Utrace_1
Vtrace_2
Wtrace_32т
J__inference_sequential_19_layer_call_and_return_conditional_losses_1737795
J__inference_sequential_19_layer_call_and_return_conditional_losses_1738233
J__inference_sequential_19_layer_call_and_return_conditional_losses_1737210
J__inference_sequential_19_layer_call_and_return_conditional_losses_1737250њ
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
 zTtrace_0zUtrace_1zVtrace_2zWtrace_3
ЎB’
"__inference__wrapped_model_1734732simple_rnn_8_input"Ш
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
л
Xiter

Ybeta_1

Zbeta_2
	[decay
\learning_rate=mш>mщ?mъ@mыAmьBmэCmюDm€EmАFmБGmВHmГImДJmЕ=vЖ>vЗ?vИ@vЙAvКBvЛCvМDvНEvОFvПGvРHvСIvТJvУ"
	optimizer
,
]serving_default"
signature_map
5
=0
>1
?2"
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
є

^states
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
В
dtrace_0
etrace_1
ftrace_2
gtrace_32Ч
.__inference_simple_rnn_8_layer_call_fn_1738244
.__inference_simple_rnn_8_layer_call_fn_1738255
.__inference_simple_rnn_8_layer_call_fn_1738266
.__inference_simple_rnn_8_layer_call_fn_1738277‘
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
 zdtrace_0zetrace_1zftrace_2zgtrace_3
о
htrace_0
itrace_1
jtrace_2
ktrace_32Г
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1738385
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1738493
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1738601
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1738709‘
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
 zhtrace_0zitrace_1zjtrace_2zktrace_3
и
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses
r_random_generator

=kernel
>recurrent_kernel
?bias"
_tf_keras_layer
 "
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
 "
trackable_list_wrapper
є

sstates
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
В
ytrace_0
ztrace_1
{trace_2
|trace_32Ч
.__inference_simple_rnn_9_layer_call_fn_1738720
.__inference_simple_rnn_9_layer_call_fn_1738731
.__inference_simple_rnn_9_layer_call_fn_1738742
.__inference_simple_rnn_9_layer_call_fn_1738753‘
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
 zytrace_0zztrace_1z{trace_2z|trace_3
р
}trace_0
~trace_1
trace_2
Аtrace_32Г
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1738863
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1738973
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1739083
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1739193‘
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
 z}trace_0z~trace_1ztrace_2zАtrace_3
п
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses
З_random_generator

@kernel
Arecurrent_kernel
Bbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
ш
Нtrace_02ў
2__inference_repeat_vector_18_layer_call_fn_1739198Ґ
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
 zНtrace_0
У
Оtrace_02ф
M__inference_repeat_vector_18_layer_call_and_return_conditional_losses_1739206Ґ
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
 zОtrace_0
5
C0
D1
E2"
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
 "
trackable_list_wrapper
њ
Пstates
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
О
Хtrace_0
Цtrace_1
Чtrace_2
Шtrace_32Ы
/__inference_simple_rnn_10_layer_call_fn_1739217
/__inference_simple_rnn_10_layer_call_fn_1739228
/__inference_simple_rnn_10_layer_call_fn_1739239
/__inference_simple_rnn_10_layer_call_fn_1739250‘
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
 zХtrace_0zЦtrace_1zЧtrace_2zШtrace_3
ъ
Щtrace_0
Ъtrace_1
Ыtrace_2
Ьtrace_32З
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1739358
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1739466
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1739574
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1739682‘
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
 zЩtrace_0zЪtrace_1zЫtrace_2zЬtrace_3
п
Э	variables
Юtrainable_variables
Яregularization_losses
†	keras_api
°__call__
+Ґ&call_and_return_all_conditional_losses
£_random_generator

Ckernel
Drecurrent_kernel
Ebias"
_tf_keras_layer
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
 "
trackable_list_wrapper
њ
§states
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
О
™trace_0
Ђtrace_1
ђtrace_2
≠trace_32Ы
/__inference_simple_rnn_11_layer_call_fn_1739693
/__inference_simple_rnn_11_layer_call_fn_1739704
/__inference_simple_rnn_11_layer_call_fn_1739715
/__inference_simple_rnn_11_layer_call_fn_1739726‘
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
 z™trace_0zЂtrace_1zђtrace_2z≠trace_3
ъ
Ѓtrace_0
ѓtrace_1
∞trace_2
±trace_32З
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1739834
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1739942
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1740050
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1740158‘
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
 zЃtrace_0zѓtrace_1z∞trace_2z±trace_3
п
≤	variables
≥trainable_variables
іregularization_losses
µ	keras_api
ґ__call__
+Ј&call_and_return_all_conditional_losses
Є_random_generator

Fkernel
Grecurrent_kernel
Hbias"
_tf_keras_layer
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
єnon_trainable_variables
Їlayers
їmetrics
 Љlayer_regularization_losses
љlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
л
Њtrace_0
њtrace_12∞
5__inference_time_distributed_18_layer_call_fn_1740167
5__inference_time_distributed_18_layer_call_fn_1740176њ
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
 zЊtrace_0zњtrace_1
°
јtrace_0
Ѕtrace_12ж
P__inference_time_distributed_18_layer_call_and_return_conditional_losses_1740197
P__inference_time_distributed_18_layer_call_and_return_conditional_losses_1740218њ
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
 zјtrace_0zЅtrace_1
Ѕ
¬	variables
√trainable_variables
ƒregularization_losses
≈	keras_api
∆__call__
+«&call_and_return_all_conditional_losses

Ikernel
Jbias"
_tf_keras_layer
7:5 2%simple_rnn_8/simple_rnn_cell_8/kernel
A:?  2/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel
1:/ 2#simple_rnn_8/simple_rnn_cell_8/bias
7:5 2%simple_rnn_9/simple_rnn_cell_9/kernel
A:?2/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel
1:/2#simple_rnn_9/simple_rnn_cell_9/bias
9:72'simple_rnn_10/simple_rnn_cell_10/kernel
C:A21simple_rnn_10/simple_rnn_cell_10/recurrent_kernel
3:12%simple_rnn_10/simple_rnn_cell_10/bias
9:7 2'simple_rnn_11/simple_rnn_cell_11/kernel
C:A  21simple_rnn_11/simple_rnn_cell_11/recurrent_kernel
3:1 2%simple_rnn_11/simple_rnn_cell_11/bias
,:* 2time_distributed_18/kernel
&:$2time_distributed_18/bias
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
(
»0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
МBЙ
/__inference_sequential_19_layer_call_fn_1736509simple_rnn_8_input"њ
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
/__inference_sequential_19_layer_call_fn_1737324inputs"њ
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
/__inference_sequential_19_layer_call_fn_1737357inputs"њ
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
/__inference_sequential_19_layer_call_fn_1737170simple_rnn_8_input"њ
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
J__inference_sequential_19_layer_call_and_return_conditional_losses_1737795inputs"њ
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
J__inference_sequential_19_layer_call_and_return_conditional_losses_1738233inputs"њ
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
J__inference_sequential_19_layer_call_and_return_conditional_losses_1737210simple_rnn_8_input"њ
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
J__inference_sequential_19_layer_call_and_return_conditional_losses_1737250simple_rnn_8_input"њ
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
%__inference_signature_wrapper_1737291simple_rnn_8_input"Ф
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
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЦBУ
.__inference_simple_rnn_8_layer_call_fn_1738244inputs/0"‘
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
.__inference_simple_rnn_8_layer_call_fn_1738255inputs/0"‘
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
.__inference_simple_rnn_8_layer_call_fn_1738266inputs"‘
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
.__inference_simple_rnn_8_layer_call_fn_1738277inputs"‘
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
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1738385inputs/0"‘
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
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1738493inputs/0"‘
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
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1738601inputs"‘
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
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1738709inputs"‘
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
=0
>1
?2"
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
…non_trainable_variables
 layers
Ћmetrics
 ћlayer_regularization_losses
Ќlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
е
ќtrace_0
ѕtrace_12™
3__inference_simple_rnn_cell_8_layer_call_fn_1740232
3__inference_simple_rnn_cell_8_layer_call_fn_1740246љ
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
 zќtrace_0zѕtrace_1
Ы
–trace_0
—trace_12а
N__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_1740263
N__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_1740280љ
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
 z–trace_0z—trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЦBУ
.__inference_simple_rnn_9_layer_call_fn_1738720inputs/0"‘
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
.__inference_simple_rnn_9_layer_call_fn_1738731inputs/0"‘
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
.__inference_simple_rnn_9_layer_call_fn_1738742inputs"‘
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
.__inference_simple_rnn_9_layer_call_fn_1738753inputs"‘
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
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1738863inputs/0"‘
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
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1738973inputs/0"‘
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
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1739083inputs"‘
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
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1739193inputs"‘
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
@0
A1
B2"
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
е
„trace_0
Ўtrace_12™
3__inference_simple_rnn_cell_9_layer_call_fn_1740294
3__inference_simple_rnn_cell_9_layer_call_fn_1740308љ
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
 z„trace_0zЎtrace_1
Ы
ўtrace_0
Џtrace_12а
N__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_1740325
N__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_1740342љ
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
 zўtrace_0zЏtrace_1
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
жBг
2__inference_repeat_vector_18_layer_call_fn_1739198inputs"Ґ
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
БBю
M__inference_repeat_vector_18_layer_call_and_return_conditional_losses_1739206inputs"Ґ
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
'
,0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЧBФ
/__inference_simple_rnn_10_layer_call_fn_1739217inputs/0"‘
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
ЧBФ
/__inference_simple_rnn_10_layer_call_fn_1739228inputs/0"‘
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
ХBТ
/__inference_simple_rnn_10_layer_call_fn_1739239inputs"‘
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
ХBТ
/__inference_simple_rnn_10_layer_call_fn_1739250inputs"‘
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
≤Bѓ
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1739358inputs/0"‘
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
≤Bѓ
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1739466inputs/0"‘
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
∞B≠
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1739574inputs"‘
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
∞B≠
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1739682inputs"‘
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
C0
D1
E2"
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
џnon_trainable_variables
№layers
Ёmetrics
 ёlayer_regularization_losses
яlayer_metrics
Э	variables
Юtrainable_variables
Яregularization_losses
°__call__
+Ґ&call_and_return_all_conditional_losses
'Ґ"call_and_return_conditional_losses"
_generic_user_object
з
аtrace_0
бtrace_12ђ
4__inference_simple_rnn_cell_10_layer_call_fn_1740356
4__inference_simple_rnn_cell_10_layer_call_fn_1740370љ
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
 zаtrace_0zбtrace_1
Э
вtrace_0
гtrace_12в
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_1740387
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_1740404љ
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
 zвtrace_0zгtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
40"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЧBФ
/__inference_simple_rnn_11_layer_call_fn_1739693inputs/0"‘
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
ЧBФ
/__inference_simple_rnn_11_layer_call_fn_1739704inputs/0"‘
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
ХBТ
/__inference_simple_rnn_11_layer_call_fn_1739715inputs"‘
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
ХBТ
/__inference_simple_rnn_11_layer_call_fn_1739726inputs"‘
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
≤Bѓ
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1739834inputs/0"‘
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
≤Bѓ
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1739942inputs/0"‘
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
∞B≠
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1740050inputs"‘
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
∞B≠
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1740158inputs"‘
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
F0
G1
H2"
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
≤	variables
≥trainable_variables
іregularization_losses
ґ__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
з
йtrace_0
кtrace_12ђ
4__inference_simple_rnn_cell_11_layer_call_fn_1740418
4__inference_simple_rnn_cell_11_layer_call_fn_1740432љ
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
 zйtrace_0zкtrace_1
Э
лtrace_0
мtrace_12в
O__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_1740449
O__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_1740466љ
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
 zлtrace_0zмtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЖBГ
5__inference_time_distributed_18_layer_call_fn_1740167inputs"њ
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
ЖBГ
5__inference_time_distributed_18_layer_call_fn_1740176inputs"њ
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
°BЮ
P__inference_time_distributed_18_layer_call_and_return_conditional_losses_1740197inputs"њ
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
°BЮ
P__inference_time_distributed_18_layer_call_and_return_conditional_losses_1740218inputs"њ
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
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
нnon_trainable_variables
оlayers
пmetrics
 рlayer_regularization_losses
сlayer_metrics
¬	variables
√trainable_variables
ƒregularization_losses
∆__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
р
тtrace_02—
*__inference_dense_21_layer_call_fn_1740475Ґ
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
 zтtrace_0
Л
уtrace_02м
E__inference_dense_21_layer_call_and_return_conditional_losses_1740485Ґ
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
 zуtrace_0
R
ф	variables
х	keras_api

цtotal

чcount"
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
3__inference_simple_rnn_cell_8_layer_call_fn_1740232inputsstates/0"љ
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
3__inference_simple_rnn_cell_8_layer_call_fn_1740246inputsstates/0"љ
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
N__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_1740263inputsstates/0"љ
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
N__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_1740280inputsstates/0"љ
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
3__inference_simple_rnn_cell_9_layer_call_fn_1740294inputsstates/0"љ
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
3__inference_simple_rnn_cell_9_layer_call_fn_1740308inputsstates/0"љ
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
N__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_1740325inputsstates/0"љ
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
N__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_1740342inputsstates/0"љ
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
НBК
4__inference_simple_rnn_cell_10_layer_call_fn_1740356inputsstates/0"љ
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
НBК
4__inference_simple_rnn_cell_10_layer_call_fn_1740370inputsstates/0"љ
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
®B•
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_1740387inputsstates/0"љ
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
®B•
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_1740404inputsstates/0"љ
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
НBК
4__inference_simple_rnn_cell_11_layer_call_fn_1740418inputsstates/0"љ
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
НBК
4__inference_simple_rnn_cell_11_layer_call_fn_1740432inputsstates/0"љ
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
®B•
O__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_1740449inputsstates/0"љ
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
®B•
O__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_1740466inputsstates/0"љ
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
*__inference_dense_21_layer_call_fn_1740475inputs"Ґ
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
E__inference_dense_21_layer_call_and_return_conditional_losses_1740485inputs"Ґ
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
0
ц0
ч1"
trackable_list_wrapper
.
ф	variables"
_generic_user_object
:  (2total
:  (2count
<:: 2,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/m
F:D  26Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/m
6:4 2*Adam/simple_rnn_8/simple_rnn_cell_8/bias/m
<:: 2,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/m
F:D26Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/m
6:42*Adam/simple_rnn_9/simple_rnn_cell_9/bias/m
>:<2.Adam/simple_rnn_10/simple_rnn_cell_10/kernel/m
H:F28Adam/simple_rnn_10/simple_rnn_cell_10/recurrent_kernel/m
8:62,Adam/simple_rnn_10/simple_rnn_cell_10/bias/m
>:< 2.Adam/simple_rnn_11/simple_rnn_cell_11/kernel/m
H:F  28Adam/simple_rnn_11/simple_rnn_cell_11/recurrent_kernel/m
8:6 2,Adam/simple_rnn_11/simple_rnn_cell_11/bias/m
1:/ 2!Adam/time_distributed_18/kernel/m
+:)2Adam/time_distributed_18/bias/m
<:: 2,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/v
F:D  26Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/v
6:4 2*Adam/simple_rnn_8/simple_rnn_cell_8/bias/v
<:: 2,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/v
F:D26Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/v
6:42*Adam/simple_rnn_9/simple_rnn_cell_9/bias/v
>:<2.Adam/simple_rnn_10/simple_rnn_cell_10/kernel/v
H:F28Adam/simple_rnn_10/simple_rnn_cell_10/recurrent_kernel/v
8:62,Adam/simple_rnn_10/simple_rnn_cell_10/bias/v
>:< 2.Adam/simple_rnn_11/simple_rnn_cell_11/kernel/v
H:F  28Adam/simple_rnn_11/simple_rnn_cell_11/recurrent_kernel/v
8:6 2,Adam/simple_rnn_11/simple_rnn_cell_11/bias/v
1:/ 2!Adam/time_distributed_18/kernel/v
+:)2Adam/time_distributed_18/bias/v«
"__inference__wrapped_model_1734732†=?>@BACEDFHGIJ?Ґ<
5Ґ2
0К-
simple_rnn_8_input€€€€€€€€€
™ "M™J
H
time_distributed_181К.
time_distributed_18€€€€€€€€€•
E__inference_dense_21_layer_call_and_return_conditional_losses_1740485\IJ/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€
Ъ }
*__inference_dense_21_layer_call_fn_1740475OIJ/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€њ
M__inference_repeat_vector_18_layer_call_and_return_conditional_losses_1739206n8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€€€€€€€€€€
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ Ч
2__inference_repeat_vector_18_layer_call_fn_1739198a8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€€€€€€€€€€
™ "%К"€€€€€€€€€€€€€€€€€€”
J__inference_sequential_19_layer_call_and_return_conditional_losses_1737210Д=?>@BACEDFHGIJGҐD
=Ґ:
0К-
simple_rnn_8_input€€€€€€€€€
p 

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ ”
J__inference_sequential_19_layer_call_and_return_conditional_losses_1737250Д=?>@BACEDFHGIJGҐD
=Ґ:
0К-
simple_rnn_8_input€€€€€€€€€
p

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ ∆
J__inference_sequential_19_layer_call_and_return_conditional_losses_1737795x=?>@BACEDFHGIJ;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ ∆
J__inference_sequential_19_layer_call_and_return_conditional_losses_1738233x=?>@BACEDFHGIJ;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ ™
/__inference_sequential_19_layer_call_fn_1736509w=?>@BACEDFHGIJGҐD
=Ґ:
0К-
simple_rnn_8_input€€€€€€€€€
p 

 
™ "К€€€€€€€€€™
/__inference_sequential_19_layer_call_fn_1737170w=?>@BACEDFHGIJGҐD
=Ґ:
0К-
simple_rnn_8_input€€€€€€€€€
p

 
™ "К€€€€€€€€€Ю
/__inference_sequential_19_layer_call_fn_1737324k=?>@BACEDFHGIJ;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€Ю
/__inference_sequential_19_layer_call_fn_1737357k=?>@BACEDFHGIJ;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€а
%__inference_signature_wrapper_1737291ґ=?>@BACEDFHGIJUҐR
Ґ 
K™H
F
simple_rnn_8_input0К-
simple_rnn_8_input€€€€€€€€€"M™J
H
time_distributed_181К.
time_distributed_18€€€€€€€€€ў
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1739358КCEDOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ ў
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1739466КCEDOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ њ
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1739574qCED?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ њ
J__inference_simple_rnn_10_layer_call_and_return_conditional_losses_1739682qCED?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ ∞
/__inference_simple_rnn_10_layer_call_fn_1739217}CEDOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "%К"€€€€€€€€€€€€€€€€€€∞
/__inference_simple_rnn_10_layer_call_fn_1739228}CEDOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "%К"€€€€€€€€€€€€€€€€€€Ч
/__inference_simple_rnn_10_layer_call_fn_1739239dCED?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€Ч
/__inference_simple_rnn_10_layer_call_fn_1739250dCED?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ "К€€€€€€€€€ў
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1739834КFHGOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ ў
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1739942КFHGOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ њ
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1740050qFHG?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ ")Ґ&
К
0€€€€€€€€€ 
Ъ њ
J__inference_simple_rnn_11_layer_call_and_return_conditional_losses_1740158qFHG?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ ")Ґ&
К
0€€€€€€€€€ 
Ъ ∞
/__inference_simple_rnn_11_layer_call_fn_1739693}FHGOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "%К"€€€€€€€€€€€€€€€€€€ ∞
/__inference_simple_rnn_11_layer_call_fn_1739704}FHGOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "%К"€€€€€€€€€€€€€€€€€€ Ч
/__inference_simple_rnn_11_layer_call_fn_1739715dFHG?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€ Ч
/__inference_simple_rnn_11_layer_call_fn_1739726dFHG?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ "К€€€€€€€€€ Ў
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1738385К=?>OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ Ў
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1738493К=?>OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ Њ
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1738601q=?>?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ ")Ґ&
К
0€€€€€€€€€ 
Ъ Њ
I__inference_simple_rnn_8_layer_call_and_return_conditional_losses_1738709q=?>?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ ")Ґ&
К
0€€€€€€€€€ 
Ъ ѓ
.__inference_simple_rnn_8_layer_call_fn_1738244}=?>OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "%К"€€€€€€€€€€€€€€€€€€ ѓ
.__inference_simple_rnn_8_layer_call_fn_1738255}=?>OҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€

 
p

 
™ "%К"€€€€€€€€€€€€€€€€€€ Ц
.__inference_simple_rnn_8_layer_call_fn_1738266d=?>?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ "К€€€€€€€€€ Ц
.__inference_simple_rnn_8_layer_call_fn_1738277d=?>?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ "К€€€€€€€€€  
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1738863}@BAOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€ 

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ  
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1738973}@BAOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€ 

 
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ї
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1739083m@BA?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€ 

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ї
I__inference_simple_rnn_9_layer_call_and_return_conditional_losses_1739193m@BA?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€ 

 
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ґ
.__inference_simple_rnn_9_layer_call_fn_1738720p@BAOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€ 

 
p 

 
™ "К€€€€€€€€€Ґ
.__inference_simple_rnn_9_layer_call_fn_1738731p@BAOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€ 

 
p

 
™ "К€€€€€€€€€Т
.__inference_simple_rnn_9_layer_call_fn_1738742`@BA?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€ 

 
p 

 
™ "К€€€€€€€€€Т
.__inference_simple_rnn_9_layer_call_fn_1738753`@BA?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€ 

 
p

 
™ "К€€€€€€€€€Л
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_1740387ЈCED\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states/0€€€€€€€€€
p 
™ "RҐO
HҐE
К
0/0€€€€€€€€€
$Ъ!
К
0/1/0€€€€€€€€€
Ъ Л
O__inference_simple_rnn_cell_10_layer_call_and_return_conditional_losses_1740404ЈCED\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states/0€€€€€€€€€
p
™ "RҐO
HҐE
К
0/0€€€€€€€€€
$Ъ!
К
0/1/0€€€€€€€€€
Ъ в
4__inference_simple_rnn_cell_10_layer_call_fn_1740356©CED\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states/0€€€€€€€€€
p 
™ "DҐA
К
0€€€€€€€€€
"Ъ
К
1/0€€€€€€€€€в
4__inference_simple_rnn_cell_10_layer_call_fn_1740370©CED\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states/0€€€€€€€€€
p
™ "DҐA
К
0€€€€€€€€€
"Ъ
К
1/0€€€€€€€€€Л
O__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_1740449ЈFHG\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states/0€€€€€€€€€ 
p 
™ "RҐO
HҐE
К
0/0€€€€€€€€€ 
$Ъ!
К
0/1/0€€€€€€€€€ 
Ъ Л
O__inference_simple_rnn_cell_11_layer_call_and_return_conditional_losses_1740466ЈFHG\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states/0€€€€€€€€€ 
p
™ "RҐO
HҐE
К
0/0€€€€€€€€€ 
$Ъ!
К
0/1/0€€€€€€€€€ 
Ъ в
4__inference_simple_rnn_cell_11_layer_call_fn_1740418©FHG\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states/0€€€€€€€€€ 
p 
™ "DҐA
К
0€€€€€€€€€ 
"Ъ
К
1/0€€€€€€€€€ в
4__inference_simple_rnn_cell_11_layer_call_fn_1740432©FHG\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states/0€€€€€€€€€ 
p
™ "DҐA
К
0€€€€€€€€€ 
"Ъ
К
1/0€€€€€€€€€ К
N__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_1740263Ј=?>\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states/0€€€€€€€€€ 
p 
™ "RҐO
HҐE
К
0/0€€€€€€€€€ 
$Ъ!
К
0/1/0€€€€€€€€€ 
Ъ К
N__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_1740280Ј=?>\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states/0€€€€€€€€€ 
p
™ "RҐO
HҐE
К
0/0€€€€€€€€€ 
$Ъ!
К
0/1/0€€€€€€€€€ 
Ъ б
3__inference_simple_rnn_cell_8_layer_call_fn_1740232©=?>\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states/0€€€€€€€€€ 
p 
™ "DҐA
К
0€€€€€€€€€ 
"Ъ
К
1/0€€€€€€€€€ б
3__inference_simple_rnn_cell_8_layer_call_fn_1740246©=?>\ҐY
RҐO
 К
inputs€€€€€€€€€
'Ґ$
"К
states/0€€€€€€€€€ 
p
™ "DҐA
К
0€€€€€€€€€ 
"Ъ
К
1/0€€€€€€€€€ К
N__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_1740325Ј@BA\ҐY
RҐO
 К
inputs€€€€€€€€€ 
'Ґ$
"К
states/0€€€€€€€€€
p 
™ "RҐO
HҐE
К
0/0€€€€€€€€€
$Ъ!
К
0/1/0€€€€€€€€€
Ъ К
N__inference_simple_rnn_cell_9_layer_call_and_return_conditional_losses_1740342Ј@BA\ҐY
RҐO
 К
inputs€€€€€€€€€ 
'Ґ$
"К
states/0€€€€€€€€€
p
™ "RҐO
HҐE
К
0/0€€€€€€€€€
$Ъ!
К
0/1/0€€€€€€€€€
Ъ б
3__inference_simple_rnn_cell_9_layer_call_fn_1740294©@BA\ҐY
RҐO
 К
inputs€€€€€€€€€ 
'Ґ$
"К
states/0€€€€€€€€€
p 
™ "DҐA
К
0€€€€€€€€€
"Ъ
К
1/0€€€€€€€€€б
3__inference_simple_rnn_cell_9_layer_call_fn_1740308©@BA\ҐY
RҐO
 К
inputs€€€€€€€€€ 
'Ґ$
"К
states/0€€€€€€€€€
p
™ "DҐA
К
0€€€€€€€€€
"Ъ
К
1/0€€€€€€€€€“
P__inference_time_distributed_18_layer_call_and_return_conditional_losses_1740197~IJDҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p 

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ “
P__inference_time_distributed_18_layer_call_and_return_conditional_losses_1740218~IJDҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ ™
5__inference_time_distributed_18_layer_call_fn_1740167qIJDҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p 

 
™ "%К"€€€€€€€€€€€€€€€€€€™
5__inference_time_distributed_18_layer_call_fn_1740176qIJDҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p

 
™ "%К"€€€€€€€€€€€€€€€€€€