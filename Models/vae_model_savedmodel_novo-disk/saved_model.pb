��
��
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
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
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
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
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
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
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
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�"serve*2.16.12v2.16.1-0-g5bc9d26649c8��
�
dense_10/biasVarHandleOp*
_output_shapes
: *

debug_namedense_10/bias/*
dtype0*
shape:*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOpdense_10/bias*
_class
loc:@Variable*
_output_shapes
:*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0
�
dense_10/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_10/kernel/*
dtype0*
shape:	�* 
shared_namedense_10/kernel
t
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes
:	�*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOpdense_10/kernel*
_class
loc:@Variable_1*
_output_shapes
:	�*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape:	�*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
j
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:	�*
dtype0
�
dense_9/biasVarHandleOp*
_output_shapes
: *

debug_namedense_9/bias/*
dtype0*
shape:�*
shared_namedense_9/bias
j
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes	
:�*
dtype0
�
%Variable_2/Initializer/ReadVariableOpReadVariableOpdense_9/bias*
_class
loc:@Variable_2*
_output_shapes	
:�*
dtype0
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape:�*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0
f
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes	
:�*
dtype0
�
dense_9/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_9/kernel/*
dtype0*
shape:
��*
shared_namedense_9/kernel
s
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel* 
_output_shapes
:
��*
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOpdense_9/kernel*
_class
loc:@Variable_3* 
_output_shapes
:
��*
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape:
��*
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
k
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3* 
_output_shapes
:
��*
dtype0
�
dense_8/biasVarHandleOp*
_output_shapes
: *

debug_namedense_8/bias/*
dtype0*
shape:�*
shared_namedense_8/bias
j
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes	
:�*
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOpdense_8/bias*
_class
loc:@Variable_4*
_output_shapes	
:�*
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape:�*
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
f
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes	
:�*
dtype0
�
dense_8/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_8/kernel/*
dtype0*
shape:
��*
shared_namedense_8/kernel
s
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel* 
_output_shapes
:
��*
dtype0
�
%Variable_5/Initializer/ReadVariableOpReadVariableOpdense_8/kernel*
_class
loc:@Variable_5* 
_output_shapes
:
��*
dtype0
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0*
shape:
��*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0
k
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5* 
_output_shapes
:
��*
dtype0
�
dense_7/biasVarHandleOp*
_output_shapes
: *

debug_namedense_7/bias/*
dtype0*
shape:�*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:�*
dtype0
�
%Variable_6/Initializer/ReadVariableOpReadVariableOpdense_7/bias*
_class
loc:@Variable_6*
_output_shapes	
:�*
dtype0
�

Variable_6VarHandleOp*
_class
loc:@Variable_6*
_output_shapes
: *

debug_nameVariable_6/*
dtype0*
shape:�*
shared_name
Variable_6
e
+Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_6*
_output_shapes
: 
e
Variable_6/AssignAssignVariableOp
Variable_6%Variable_6/Initializer/ReadVariableOp*
dtype0
f
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes	
:�*
dtype0
�
dense_7/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_7/kernel/*
dtype0*
shape:
��*
shared_namedense_7/kernel
s
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel* 
_output_shapes
:
��*
dtype0
�
%Variable_7/Initializer/ReadVariableOpReadVariableOpdense_7/kernel*
_class
loc:@Variable_7* 
_output_shapes
:
��*
dtype0
�

Variable_7VarHandleOp*
_class
loc:@Variable_7*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape:
��*
shared_name
Variable_7
e
+Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_7*
_output_shapes
: 
e
Variable_7/AssignAssignVariableOp
Variable_7%Variable_7/Initializer/ReadVariableOp*
dtype0
k
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7* 
_output_shapes
:
��*
dtype0
�
dense_6/biasVarHandleOp*
_output_shapes
: *

debug_namedense_6/bias/*
dtype0*
shape:�*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:�*
dtype0
�
%Variable_8/Initializer/ReadVariableOpReadVariableOpdense_6/bias*
_class
loc:@Variable_8*
_output_shapes	
:�*
dtype0
�

Variable_8VarHandleOp*
_class
loc:@Variable_8*
_output_shapes
: *

debug_nameVariable_8/*
dtype0*
shape:�*
shared_name
Variable_8
e
+Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_8*
_output_shapes
: 
e
Variable_8/AssignAssignVariableOp
Variable_8%Variable_8/Initializer/ReadVariableOp*
dtype0
f
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes	
:�*
dtype0
�
dense_6/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_6/kernel/*
dtype0*
shape:	d�*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	d�*
dtype0
�
%Variable_9/Initializer/ReadVariableOpReadVariableOpdense_6/kernel*
_class
loc:@Variable_9*
_output_shapes
:	d�*
dtype0
�

Variable_9VarHandleOp*
_class
loc:@Variable_9*
_output_shapes
: *

debug_nameVariable_9/*
dtype0*
shape:	d�*
shared_name
Variable_9
e
+Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_9*
_output_shapes
: 
e
Variable_9/AssignAssignVariableOp
Variable_9%Variable_9/Initializer/ReadVariableOp*
dtype0
j
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes
:	d�*
dtype0
�
adam/dense_10_bias_velocityVarHandleOp*
_output_shapes
: *,

debug_nameadam/dense_10_bias_velocity/*
dtype0*
shape:*,
shared_nameadam/dense_10_bias_velocity
�
/adam/dense_10_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_10_bias_velocity*
_output_shapes
:*
dtype0
�
&Variable_10/Initializer/ReadVariableOpReadVariableOpadam/dense_10_bias_velocity*
_class
loc:@Variable_10*
_output_shapes
:*
dtype0
�
Variable_10VarHandleOp*
_class
loc:@Variable_10*
_output_shapes
: *

debug_nameVariable_10/*
dtype0*
shape:*
shared_nameVariable_10
g
,Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_10*
_output_shapes
: 
h
Variable_10/AssignAssignVariableOpVariable_10&Variable_10/Initializer/ReadVariableOp*
dtype0
g
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*
_output_shapes
:*
dtype0
�
adam/dense_10_bias_momentumVarHandleOp*
_output_shapes
: *,

debug_nameadam/dense_10_bias_momentum/*
dtype0*
shape:*,
shared_nameadam/dense_10_bias_momentum
�
/adam/dense_10_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_10_bias_momentum*
_output_shapes
:*
dtype0
�
&Variable_11/Initializer/ReadVariableOpReadVariableOpadam/dense_10_bias_momentum*
_class
loc:@Variable_11*
_output_shapes
:*
dtype0
�
Variable_11VarHandleOp*
_class
loc:@Variable_11*
_output_shapes
: *

debug_nameVariable_11/*
dtype0*
shape:*
shared_nameVariable_11
g
,Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_11*
_output_shapes
: 
h
Variable_11/AssignAssignVariableOpVariable_11&Variable_11/Initializer/ReadVariableOp*
dtype0
g
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*
_output_shapes
:*
dtype0
�
adam/dense_10_kernel_velocityVarHandleOp*
_output_shapes
: *.

debug_name adam/dense_10_kernel_velocity/*
dtype0*
shape:	�*.
shared_nameadam/dense_10_kernel_velocity
�
1adam/dense_10_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_10_kernel_velocity*
_output_shapes
:	�*
dtype0
�
&Variable_12/Initializer/ReadVariableOpReadVariableOpadam/dense_10_kernel_velocity*
_class
loc:@Variable_12*
_output_shapes
:	�*
dtype0
�
Variable_12VarHandleOp*
_class
loc:@Variable_12*
_output_shapes
: *

debug_nameVariable_12/*
dtype0*
shape:	�*
shared_nameVariable_12
g
,Variable_12/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_12*
_output_shapes
: 
h
Variable_12/AssignAssignVariableOpVariable_12&Variable_12/Initializer/ReadVariableOp*
dtype0
l
Variable_12/Read/ReadVariableOpReadVariableOpVariable_12*
_output_shapes
:	�*
dtype0
�
adam/dense_10_kernel_momentumVarHandleOp*
_output_shapes
: *.

debug_name adam/dense_10_kernel_momentum/*
dtype0*
shape:	�*.
shared_nameadam/dense_10_kernel_momentum
�
1adam/dense_10_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_10_kernel_momentum*
_output_shapes
:	�*
dtype0
�
&Variable_13/Initializer/ReadVariableOpReadVariableOpadam/dense_10_kernel_momentum*
_class
loc:@Variable_13*
_output_shapes
:	�*
dtype0
�
Variable_13VarHandleOp*
_class
loc:@Variable_13*
_output_shapes
: *

debug_nameVariable_13/*
dtype0*
shape:	�*
shared_nameVariable_13
g
,Variable_13/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_13*
_output_shapes
: 
h
Variable_13/AssignAssignVariableOpVariable_13&Variable_13/Initializer/ReadVariableOp*
dtype0
l
Variable_13/Read/ReadVariableOpReadVariableOpVariable_13*
_output_shapes
:	�*
dtype0
�
adam/dense_9_bias_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_9_bias_velocity/*
dtype0*
shape:�*+
shared_nameadam/dense_9_bias_velocity
�
.adam/dense_9_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_9_bias_velocity*
_output_shapes	
:�*
dtype0
�
&Variable_14/Initializer/ReadVariableOpReadVariableOpadam/dense_9_bias_velocity*
_class
loc:@Variable_14*
_output_shapes	
:�*
dtype0
�
Variable_14VarHandleOp*
_class
loc:@Variable_14*
_output_shapes
: *

debug_nameVariable_14/*
dtype0*
shape:�*
shared_nameVariable_14
g
,Variable_14/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_14*
_output_shapes
: 
h
Variable_14/AssignAssignVariableOpVariable_14&Variable_14/Initializer/ReadVariableOp*
dtype0
h
Variable_14/Read/ReadVariableOpReadVariableOpVariable_14*
_output_shapes	
:�*
dtype0
�
adam/dense_9_bias_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_9_bias_momentum/*
dtype0*
shape:�*+
shared_nameadam/dense_9_bias_momentum
�
.adam/dense_9_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_9_bias_momentum*
_output_shapes	
:�*
dtype0
�
&Variable_15/Initializer/ReadVariableOpReadVariableOpadam/dense_9_bias_momentum*
_class
loc:@Variable_15*
_output_shapes	
:�*
dtype0
�
Variable_15VarHandleOp*
_class
loc:@Variable_15*
_output_shapes
: *

debug_nameVariable_15/*
dtype0*
shape:�*
shared_nameVariable_15
g
,Variable_15/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_15*
_output_shapes
: 
h
Variable_15/AssignAssignVariableOpVariable_15&Variable_15/Initializer/ReadVariableOp*
dtype0
h
Variable_15/Read/ReadVariableOpReadVariableOpVariable_15*
_output_shapes	
:�*
dtype0
�
adam/dense_9_kernel_velocityVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_9_kernel_velocity/*
dtype0*
shape:
��*-
shared_nameadam/dense_9_kernel_velocity
�
0adam/dense_9_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_9_kernel_velocity* 
_output_shapes
:
��*
dtype0
�
&Variable_16/Initializer/ReadVariableOpReadVariableOpadam/dense_9_kernel_velocity*
_class
loc:@Variable_16* 
_output_shapes
:
��*
dtype0
�
Variable_16VarHandleOp*
_class
loc:@Variable_16*
_output_shapes
: *

debug_nameVariable_16/*
dtype0*
shape:
��*
shared_nameVariable_16
g
,Variable_16/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_16*
_output_shapes
: 
h
Variable_16/AssignAssignVariableOpVariable_16&Variable_16/Initializer/ReadVariableOp*
dtype0
m
Variable_16/Read/ReadVariableOpReadVariableOpVariable_16* 
_output_shapes
:
��*
dtype0
�
adam/dense_9_kernel_momentumVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_9_kernel_momentum/*
dtype0*
shape:
��*-
shared_nameadam/dense_9_kernel_momentum
�
0adam/dense_9_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_9_kernel_momentum* 
_output_shapes
:
��*
dtype0
�
&Variable_17/Initializer/ReadVariableOpReadVariableOpadam/dense_9_kernel_momentum*
_class
loc:@Variable_17* 
_output_shapes
:
��*
dtype0
�
Variable_17VarHandleOp*
_class
loc:@Variable_17*
_output_shapes
: *

debug_nameVariable_17/*
dtype0*
shape:
��*
shared_nameVariable_17
g
,Variable_17/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_17*
_output_shapes
: 
h
Variable_17/AssignAssignVariableOpVariable_17&Variable_17/Initializer/ReadVariableOp*
dtype0
m
Variable_17/Read/ReadVariableOpReadVariableOpVariable_17* 
_output_shapes
:
��*
dtype0
�
adam/dense_8_bias_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_8_bias_velocity/*
dtype0*
shape:�*+
shared_nameadam/dense_8_bias_velocity
�
.adam/dense_8_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_8_bias_velocity*
_output_shapes	
:�*
dtype0
�
&Variable_18/Initializer/ReadVariableOpReadVariableOpadam/dense_8_bias_velocity*
_class
loc:@Variable_18*
_output_shapes	
:�*
dtype0
�
Variable_18VarHandleOp*
_class
loc:@Variable_18*
_output_shapes
: *

debug_nameVariable_18/*
dtype0*
shape:�*
shared_nameVariable_18
g
,Variable_18/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_18*
_output_shapes
: 
h
Variable_18/AssignAssignVariableOpVariable_18&Variable_18/Initializer/ReadVariableOp*
dtype0
h
Variable_18/Read/ReadVariableOpReadVariableOpVariable_18*
_output_shapes	
:�*
dtype0
�
adam/dense_8_bias_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_8_bias_momentum/*
dtype0*
shape:�*+
shared_nameadam/dense_8_bias_momentum
�
.adam/dense_8_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_8_bias_momentum*
_output_shapes	
:�*
dtype0
�
&Variable_19/Initializer/ReadVariableOpReadVariableOpadam/dense_8_bias_momentum*
_class
loc:@Variable_19*
_output_shapes	
:�*
dtype0
�
Variable_19VarHandleOp*
_class
loc:@Variable_19*
_output_shapes
: *

debug_nameVariable_19/*
dtype0*
shape:�*
shared_nameVariable_19
g
,Variable_19/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_19*
_output_shapes
: 
h
Variable_19/AssignAssignVariableOpVariable_19&Variable_19/Initializer/ReadVariableOp*
dtype0
h
Variable_19/Read/ReadVariableOpReadVariableOpVariable_19*
_output_shapes	
:�*
dtype0
�
adam/dense_8_kernel_velocityVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_8_kernel_velocity/*
dtype0*
shape:
��*-
shared_nameadam/dense_8_kernel_velocity
�
0adam/dense_8_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_8_kernel_velocity* 
_output_shapes
:
��*
dtype0
�
&Variable_20/Initializer/ReadVariableOpReadVariableOpadam/dense_8_kernel_velocity*
_class
loc:@Variable_20* 
_output_shapes
:
��*
dtype0
�
Variable_20VarHandleOp*
_class
loc:@Variable_20*
_output_shapes
: *

debug_nameVariable_20/*
dtype0*
shape:
��*
shared_nameVariable_20
g
,Variable_20/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_20*
_output_shapes
: 
h
Variable_20/AssignAssignVariableOpVariable_20&Variable_20/Initializer/ReadVariableOp*
dtype0
m
Variable_20/Read/ReadVariableOpReadVariableOpVariable_20* 
_output_shapes
:
��*
dtype0
�
adam/dense_8_kernel_momentumVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_8_kernel_momentum/*
dtype0*
shape:
��*-
shared_nameadam/dense_8_kernel_momentum
�
0adam/dense_8_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_8_kernel_momentum* 
_output_shapes
:
��*
dtype0
�
&Variable_21/Initializer/ReadVariableOpReadVariableOpadam/dense_8_kernel_momentum*
_class
loc:@Variable_21* 
_output_shapes
:
��*
dtype0
�
Variable_21VarHandleOp*
_class
loc:@Variable_21*
_output_shapes
: *

debug_nameVariable_21/*
dtype0*
shape:
��*
shared_nameVariable_21
g
,Variable_21/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_21*
_output_shapes
: 
h
Variable_21/AssignAssignVariableOpVariable_21&Variable_21/Initializer/ReadVariableOp*
dtype0
m
Variable_21/Read/ReadVariableOpReadVariableOpVariable_21* 
_output_shapes
:
��*
dtype0
�
adam/dense_7_bias_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_7_bias_velocity/*
dtype0*
shape:�*+
shared_nameadam/dense_7_bias_velocity
�
.adam/dense_7_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_7_bias_velocity*
_output_shapes	
:�*
dtype0
�
&Variable_22/Initializer/ReadVariableOpReadVariableOpadam/dense_7_bias_velocity*
_class
loc:@Variable_22*
_output_shapes	
:�*
dtype0
�
Variable_22VarHandleOp*
_class
loc:@Variable_22*
_output_shapes
: *

debug_nameVariable_22/*
dtype0*
shape:�*
shared_nameVariable_22
g
,Variable_22/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_22*
_output_shapes
: 
h
Variable_22/AssignAssignVariableOpVariable_22&Variable_22/Initializer/ReadVariableOp*
dtype0
h
Variable_22/Read/ReadVariableOpReadVariableOpVariable_22*
_output_shapes	
:�*
dtype0
�
adam/dense_7_bias_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_7_bias_momentum/*
dtype0*
shape:�*+
shared_nameadam/dense_7_bias_momentum
�
.adam/dense_7_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_7_bias_momentum*
_output_shapes	
:�*
dtype0
�
&Variable_23/Initializer/ReadVariableOpReadVariableOpadam/dense_7_bias_momentum*
_class
loc:@Variable_23*
_output_shapes	
:�*
dtype0
�
Variable_23VarHandleOp*
_class
loc:@Variable_23*
_output_shapes
: *

debug_nameVariable_23/*
dtype0*
shape:�*
shared_nameVariable_23
g
,Variable_23/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_23*
_output_shapes
: 
h
Variable_23/AssignAssignVariableOpVariable_23&Variable_23/Initializer/ReadVariableOp*
dtype0
h
Variable_23/Read/ReadVariableOpReadVariableOpVariable_23*
_output_shapes	
:�*
dtype0
�
adam/dense_7_kernel_velocityVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_7_kernel_velocity/*
dtype0*
shape:
��*-
shared_nameadam/dense_7_kernel_velocity
�
0adam/dense_7_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_7_kernel_velocity* 
_output_shapes
:
��*
dtype0
�
&Variable_24/Initializer/ReadVariableOpReadVariableOpadam/dense_7_kernel_velocity*
_class
loc:@Variable_24* 
_output_shapes
:
��*
dtype0
�
Variable_24VarHandleOp*
_class
loc:@Variable_24*
_output_shapes
: *

debug_nameVariable_24/*
dtype0*
shape:
��*
shared_nameVariable_24
g
,Variable_24/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_24*
_output_shapes
: 
h
Variable_24/AssignAssignVariableOpVariable_24&Variable_24/Initializer/ReadVariableOp*
dtype0
m
Variable_24/Read/ReadVariableOpReadVariableOpVariable_24* 
_output_shapes
:
��*
dtype0
�
adam/dense_7_kernel_momentumVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_7_kernel_momentum/*
dtype0*
shape:
��*-
shared_nameadam/dense_7_kernel_momentum
�
0adam/dense_7_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_7_kernel_momentum* 
_output_shapes
:
��*
dtype0
�
&Variable_25/Initializer/ReadVariableOpReadVariableOpadam/dense_7_kernel_momentum*
_class
loc:@Variable_25* 
_output_shapes
:
��*
dtype0
�
Variable_25VarHandleOp*
_class
loc:@Variable_25*
_output_shapes
: *

debug_nameVariable_25/*
dtype0*
shape:
��*
shared_nameVariable_25
g
,Variable_25/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_25*
_output_shapes
: 
h
Variable_25/AssignAssignVariableOpVariable_25&Variable_25/Initializer/ReadVariableOp*
dtype0
m
Variable_25/Read/ReadVariableOpReadVariableOpVariable_25* 
_output_shapes
:
��*
dtype0
�
adam/dense_6_bias_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_6_bias_velocity/*
dtype0*
shape:�*+
shared_nameadam/dense_6_bias_velocity
�
.adam/dense_6_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_6_bias_velocity*
_output_shapes	
:�*
dtype0
�
&Variable_26/Initializer/ReadVariableOpReadVariableOpadam/dense_6_bias_velocity*
_class
loc:@Variable_26*
_output_shapes	
:�*
dtype0
�
Variable_26VarHandleOp*
_class
loc:@Variable_26*
_output_shapes
: *

debug_nameVariable_26/*
dtype0*
shape:�*
shared_nameVariable_26
g
,Variable_26/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_26*
_output_shapes
: 
h
Variable_26/AssignAssignVariableOpVariable_26&Variable_26/Initializer/ReadVariableOp*
dtype0
h
Variable_26/Read/ReadVariableOpReadVariableOpVariable_26*
_output_shapes	
:�*
dtype0
�
adam/dense_6_bias_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_6_bias_momentum/*
dtype0*
shape:�*+
shared_nameadam/dense_6_bias_momentum
�
.adam/dense_6_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_6_bias_momentum*
_output_shapes	
:�*
dtype0
�
&Variable_27/Initializer/ReadVariableOpReadVariableOpadam/dense_6_bias_momentum*
_class
loc:@Variable_27*
_output_shapes	
:�*
dtype0
�
Variable_27VarHandleOp*
_class
loc:@Variable_27*
_output_shapes
: *

debug_nameVariable_27/*
dtype0*
shape:�*
shared_nameVariable_27
g
,Variable_27/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_27*
_output_shapes
: 
h
Variable_27/AssignAssignVariableOpVariable_27&Variable_27/Initializer/ReadVariableOp*
dtype0
h
Variable_27/Read/ReadVariableOpReadVariableOpVariable_27*
_output_shapes	
:�*
dtype0
�
adam/dense_6_kernel_velocityVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_6_kernel_velocity/*
dtype0*
shape:	d�*-
shared_nameadam/dense_6_kernel_velocity
�
0adam/dense_6_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_6_kernel_velocity*
_output_shapes
:	d�*
dtype0
�
&Variable_28/Initializer/ReadVariableOpReadVariableOpadam/dense_6_kernel_velocity*
_class
loc:@Variable_28*
_output_shapes
:	d�*
dtype0
�
Variable_28VarHandleOp*
_class
loc:@Variable_28*
_output_shapes
: *

debug_nameVariable_28/*
dtype0*
shape:	d�*
shared_nameVariable_28
g
,Variable_28/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_28*
_output_shapes
: 
h
Variable_28/AssignAssignVariableOpVariable_28&Variable_28/Initializer/ReadVariableOp*
dtype0
l
Variable_28/Read/ReadVariableOpReadVariableOpVariable_28*
_output_shapes
:	d�*
dtype0
�
adam/dense_6_kernel_momentumVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_6_kernel_momentum/*
dtype0*
shape:	d�*-
shared_nameadam/dense_6_kernel_momentum
�
0adam/dense_6_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_6_kernel_momentum*
_output_shapes
:	d�*
dtype0
�
&Variable_29/Initializer/ReadVariableOpReadVariableOpadam/dense_6_kernel_momentum*
_class
loc:@Variable_29*
_output_shapes
:	d�*
dtype0
�
Variable_29VarHandleOp*
_class
loc:@Variable_29*
_output_shapes
: *

debug_nameVariable_29/*
dtype0*
shape:	d�*
shared_nameVariable_29
g
,Variable_29/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_29*
_output_shapes
: 
h
Variable_29/AssignAssignVariableOpVariable_29&Variable_29/Initializer/ReadVariableOp*
dtype0
l
Variable_29/Read/ReadVariableOpReadVariableOpVariable_29*
_output_shapes
:	d�*
dtype0
�
adam/dense_5_bias_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_5_bias_velocity/*
dtype0*
shape:d*+
shared_nameadam/dense_5_bias_velocity
�
.adam/dense_5_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_5_bias_velocity*
_output_shapes
:d*
dtype0
�
&Variable_30/Initializer/ReadVariableOpReadVariableOpadam/dense_5_bias_velocity*
_class
loc:@Variable_30*
_output_shapes
:d*
dtype0
�
Variable_30VarHandleOp*
_class
loc:@Variable_30*
_output_shapes
: *

debug_nameVariable_30/*
dtype0*
shape:d*
shared_nameVariable_30
g
,Variable_30/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_30*
_output_shapes
: 
h
Variable_30/AssignAssignVariableOpVariable_30&Variable_30/Initializer/ReadVariableOp*
dtype0
g
Variable_30/Read/ReadVariableOpReadVariableOpVariable_30*
_output_shapes
:d*
dtype0
�
adam/dense_5_bias_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_5_bias_momentum/*
dtype0*
shape:d*+
shared_nameadam/dense_5_bias_momentum
�
.adam/dense_5_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_5_bias_momentum*
_output_shapes
:d*
dtype0
�
&Variable_31/Initializer/ReadVariableOpReadVariableOpadam/dense_5_bias_momentum*
_class
loc:@Variable_31*
_output_shapes
:d*
dtype0
�
Variable_31VarHandleOp*
_class
loc:@Variable_31*
_output_shapes
: *

debug_nameVariable_31/*
dtype0*
shape:d*
shared_nameVariable_31
g
,Variable_31/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_31*
_output_shapes
: 
h
Variable_31/AssignAssignVariableOpVariable_31&Variable_31/Initializer/ReadVariableOp*
dtype0
g
Variable_31/Read/ReadVariableOpReadVariableOpVariable_31*
_output_shapes
:d*
dtype0
�
adam/dense_5_kernel_velocityVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_5_kernel_velocity/*
dtype0*
shape:	�d*-
shared_nameadam/dense_5_kernel_velocity
�
0adam/dense_5_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_5_kernel_velocity*
_output_shapes
:	�d*
dtype0
�
&Variable_32/Initializer/ReadVariableOpReadVariableOpadam/dense_5_kernel_velocity*
_class
loc:@Variable_32*
_output_shapes
:	�d*
dtype0
�
Variable_32VarHandleOp*
_class
loc:@Variable_32*
_output_shapes
: *

debug_nameVariable_32/*
dtype0*
shape:	�d*
shared_nameVariable_32
g
,Variable_32/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_32*
_output_shapes
: 
h
Variable_32/AssignAssignVariableOpVariable_32&Variable_32/Initializer/ReadVariableOp*
dtype0
l
Variable_32/Read/ReadVariableOpReadVariableOpVariable_32*
_output_shapes
:	�d*
dtype0
�
adam/dense_5_kernel_momentumVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_5_kernel_momentum/*
dtype0*
shape:	�d*-
shared_nameadam/dense_5_kernel_momentum
�
0adam/dense_5_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_5_kernel_momentum*
_output_shapes
:	�d*
dtype0
�
&Variable_33/Initializer/ReadVariableOpReadVariableOpadam/dense_5_kernel_momentum*
_class
loc:@Variable_33*
_output_shapes
:	�d*
dtype0
�
Variable_33VarHandleOp*
_class
loc:@Variable_33*
_output_shapes
: *

debug_nameVariable_33/*
dtype0*
shape:	�d*
shared_nameVariable_33
g
,Variable_33/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_33*
_output_shapes
: 
h
Variable_33/AssignAssignVariableOpVariable_33&Variable_33/Initializer/ReadVariableOp*
dtype0
l
Variable_33/Read/ReadVariableOpReadVariableOpVariable_33*
_output_shapes
:	�d*
dtype0
�
adam/dense_4_bias_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_4_bias_velocity/*
dtype0*
shape:d*+
shared_nameadam/dense_4_bias_velocity
�
.adam/dense_4_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_4_bias_velocity*
_output_shapes
:d*
dtype0
�
&Variable_34/Initializer/ReadVariableOpReadVariableOpadam/dense_4_bias_velocity*
_class
loc:@Variable_34*
_output_shapes
:d*
dtype0
�
Variable_34VarHandleOp*
_class
loc:@Variable_34*
_output_shapes
: *

debug_nameVariable_34/*
dtype0*
shape:d*
shared_nameVariable_34
g
,Variable_34/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_34*
_output_shapes
: 
h
Variable_34/AssignAssignVariableOpVariable_34&Variable_34/Initializer/ReadVariableOp*
dtype0
g
Variable_34/Read/ReadVariableOpReadVariableOpVariable_34*
_output_shapes
:d*
dtype0
�
adam/dense_4_bias_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_4_bias_momentum/*
dtype0*
shape:d*+
shared_nameadam/dense_4_bias_momentum
�
.adam/dense_4_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_4_bias_momentum*
_output_shapes
:d*
dtype0
�
&Variable_35/Initializer/ReadVariableOpReadVariableOpadam/dense_4_bias_momentum*
_class
loc:@Variable_35*
_output_shapes
:d*
dtype0
�
Variable_35VarHandleOp*
_class
loc:@Variable_35*
_output_shapes
: *

debug_nameVariable_35/*
dtype0*
shape:d*
shared_nameVariable_35
g
,Variable_35/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_35*
_output_shapes
: 
h
Variable_35/AssignAssignVariableOpVariable_35&Variable_35/Initializer/ReadVariableOp*
dtype0
g
Variable_35/Read/ReadVariableOpReadVariableOpVariable_35*
_output_shapes
:d*
dtype0
�
adam/dense_4_kernel_velocityVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_4_kernel_velocity/*
dtype0*
shape:	�d*-
shared_nameadam/dense_4_kernel_velocity
�
0adam/dense_4_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_4_kernel_velocity*
_output_shapes
:	�d*
dtype0
�
&Variable_36/Initializer/ReadVariableOpReadVariableOpadam/dense_4_kernel_velocity*
_class
loc:@Variable_36*
_output_shapes
:	�d*
dtype0
�
Variable_36VarHandleOp*
_class
loc:@Variable_36*
_output_shapes
: *

debug_nameVariable_36/*
dtype0*
shape:	�d*
shared_nameVariable_36
g
,Variable_36/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_36*
_output_shapes
: 
h
Variable_36/AssignAssignVariableOpVariable_36&Variable_36/Initializer/ReadVariableOp*
dtype0
l
Variable_36/Read/ReadVariableOpReadVariableOpVariable_36*
_output_shapes
:	�d*
dtype0
�
adam/dense_4_kernel_momentumVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_4_kernel_momentum/*
dtype0*
shape:	�d*-
shared_nameadam/dense_4_kernel_momentum
�
0adam/dense_4_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_4_kernel_momentum*
_output_shapes
:	�d*
dtype0
�
&Variable_37/Initializer/ReadVariableOpReadVariableOpadam/dense_4_kernel_momentum*
_class
loc:@Variable_37*
_output_shapes
:	�d*
dtype0
�
Variable_37VarHandleOp*
_class
loc:@Variable_37*
_output_shapes
: *

debug_nameVariable_37/*
dtype0*
shape:	�d*
shared_nameVariable_37
g
,Variable_37/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_37*
_output_shapes
: 
h
Variable_37/AssignAssignVariableOpVariable_37&Variable_37/Initializer/ReadVariableOp*
dtype0
l
Variable_37/Read/ReadVariableOpReadVariableOpVariable_37*
_output_shapes
:	�d*
dtype0
�
adam/dense_3_bias_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_3_bias_velocity/*
dtype0*
shape:�*+
shared_nameadam/dense_3_bias_velocity
�
.adam/dense_3_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_3_bias_velocity*
_output_shapes	
:�*
dtype0
�
&Variable_38/Initializer/ReadVariableOpReadVariableOpadam/dense_3_bias_velocity*
_class
loc:@Variable_38*
_output_shapes	
:�*
dtype0
�
Variable_38VarHandleOp*
_class
loc:@Variable_38*
_output_shapes
: *

debug_nameVariable_38/*
dtype0*
shape:�*
shared_nameVariable_38
g
,Variable_38/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_38*
_output_shapes
: 
h
Variable_38/AssignAssignVariableOpVariable_38&Variable_38/Initializer/ReadVariableOp*
dtype0
h
Variable_38/Read/ReadVariableOpReadVariableOpVariable_38*
_output_shapes	
:�*
dtype0
�
adam/dense_3_bias_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_3_bias_momentum/*
dtype0*
shape:�*+
shared_nameadam/dense_3_bias_momentum
�
.adam/dense_3_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_3_bias_momentum*
_output_shapes	
:�*
dtype0
�
&Variable_39/Initializer/ReadVariableOpReadVariableOpadam/dense_3_bias_momentum*
_class
loc:@Variable_39*
_output_shapes	
:�*
dtype0
�
Variable_39VarHandleOp*
_class
loc:@Variable_39*
_output_shapes
: *

debug_nameVariable_39/*
dtype0*
shape:�*
shared_nameVariable_39
g
,Variable_39/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_39*
_output_shapes
: 
h
Variable_39/AssignAssignVariableOpVariable_39&Variable_39/Initializer/ReadVariableOp*
dtype0
h
Variable_39/Read/ReadVariableOpReadVariableOpVariable_39*
_output_shapes	
:�*
dtype0
�
adam/dense_3_kernel_velocityVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_3_kernel_velocity/*
dtype0*
shape:
��*-
shared_nameadam/dense_3_kernel_velocity
�
0adam/dense_3_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_3_kernel_velocity* 
_output_shapes
:
��*
dtype0
�
&Variable_40/Initializer/ReadVariableOpReadVariableOpadam/dense_3_kernel_velocity*
_class
loc:@Variable_40* 
_output_shapes
:
��*
dtype0
�
Variable_40VarHandleOp*
_class
loc:@Variable_40*
_output_shapes
: *

debug_nameVariable_40/*
dtype0*
shape:
��*
shared_nameVariable_40
g
,Variable_40/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_40*
_output_shapes
: 
h
Variable_40/AssignAssignVariableOpVariable_40&Variable_40/Initializer/ReadVariableOp*
dtype0
m
Variable_40/Read/ReadVariableOpReadVariableOpVariable_40* 
_output_shapes
:
��*
dtype0
�
adam/dense_3_kernel_momentumVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_3_kernel_momentum/*
dtype0*
shape:
��*-
shared_nameadam/dense_3_kernel_momentum
�
0adam/dense_3_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_3_kernel_momentum* 
_output_shapes
:
��*
dtype0
�
&Variable_41/Initializer/ReadVariableOpReadVariableOpadam/dense_3_kernel_momentum*
_class
loc:@Variable_41* 
_output_shapes
:
��*
dtype0
�
Variable_41VarHandleOp*
_class
loc:@Variable_41*
_output_shapes
: *

debug_nameVariable_41/*
dtype0*
shape:
��*
shared_nameVariable_41
g
,Variable_41/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_41*
_output_shapes
: 
h
Variable_41/AssignAssignVariableOpVariable_41&Variable_41/Initializer/ReadVariableOp*
dtype0
m
Variable_41/Read/ReadVariableOpReadVariableOpVariable_41* 
_output_shapes
:
��*
dtype0
�
adam/dense_2_bias_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_2_bias_velocity/*
dtype0*
shape:�*+
shared_nameadam/dense_2_bias_velocity
�
.adam/dense_2_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_2_bias_velocity*
_output_shapes	
:�*
dtype0
�
&Variable_42/Initializer/ReadVariableOpReadVariableOpadam/dense_2_bias_velocity*
_class
loc:@Variable_42*
_output_shapes	
:�*
dtype0
�
Variable_42VarHandleOp*
_class
loc:@Variable_42*
_output_shapes
: *

debug_nameVariable_42/*
dtype0*
shape:�*
shared_nameVariable_42
g
,Variable_42/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_42*
_output_shapes
: 
h
Variable_42/AssignAssignVariableOpVariable_42&Variable_42/Initializer/ReadVariableOp*
dtype0
h
Variable_42/Read/ReadVariableOpReadVariableOpVariable_42*
_output_shapes	
:�*
dtype0
�
adam/dense_2_bias_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_2_bias_momentum/*
dtype0*
shape:�*+
shared_nameadam/dense_2_bias_momentum
�
.adam/dense_2_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_2_bias_momentum*
_output_shapes	
:�*
dtype0
�
&Variable_43/Initializer/ReadVariableOpReadVariableOpadam/dense_2_bias_momentum*
_class
loc:@Variable_43*
_output_shapes	
:�*
dtype0
�
Variable_43VarHandleOp*
_class
loc:@Variable_43*
_output_shapes
: *

debug_nameVariable_43/*
dtype0*
shape:�*
shared_nameVariable_43
g
,Variable_43/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_43*
_output_shapes
: 
h
Variable_43/AssignAssignVariableOpVariable_43&Variable_43/Initializer/ReadVariableOp*
dtype0
h
Variable_43/Read/ReadVariableOpReadVariableOpVariable_43*
_output_shapes	
:�*
dtype0
�
adam/dense_2_kernel_velocityVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_2_kernel_velocity/*
dtype0*
shape:
��*-
shared_nameadam/dense_2_kernel_velocity
�
0adam/dense_2_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_2_kernel_velocity* 
_output_shapes
:
��*
dtype0
�
&Variable_44/Initializer/ReadVariableOpReadVariableOpadam/dense_2_kernel_velocity*
_class
loc:@Variable_44* 
_output_shapes
:
��*
dtype0
�
Variable_44VarHandleOp*
_class
loc:@Variable_44*
_output_shapes
: *

debug_nameVariable_44/*
dtype0*
shape:
��*
shared_nameVariable_44
g
,Variable_44/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_44*
_output_shapes
: 
h
Variable_44/AssignAssignVariableOpVariable_44&Variable_44/Initializer/ReadVariableOp*
dtype0
m
Variable_44/Read/ReadVariableOpReadVariableOpVariable_44* 
_output_shapes
:
��*
dtype0
�
adam/dense_2_kernel_momentumVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_2_kernel_momentum/*
dtype0*
shape:
��*-
shared_nameadam/dense_2_kernel_momentum
�
0adam/dense_2_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_2_kernel_momentum* 
_output_shapes
:
��*
dtype0
�
&Variable_45/Initializer/ReadVariableOpReadVariableOpadam/dense_2_kernel_momentum*
_class
loc:@Variable_45* 
_output_shapes
:
��*
dtype0
�
Variable_45VarHandleOp*
_class
loc:@Variable_45*
_output_shapes
: *

debug_nameVariable_45/*
dtype0*
shape:
��*
shared_nameVariable_45
g
,Variable_45/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_45*
_output_shapes
: 
h
Variable_45/AssignAssignVariableOpVariable_45&Variable_45/Initializer/ReadVariableOp*
dtype0
m
Variable_45/Read/ReadVariableOpReadVariableOpVariable_45* 
_output_shapes
:
��*
dtype0
�
adam/dense_1_bias_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_1_bias_velocity/*
dtype0*
shape:�*+
shared_nameadam/dense_1_bias_velocity
�
.adam/dense_1_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_1_bias_velocity*
_output_shapes	
:�*
dtype0
�
&Variable_46/Initializer/ReadVariableOpReadVariableOpadam/dense_1_bias_velocity*
_class
loc:@Variable_46*
_output_shapes	
:�*
dtype0
�
Variable_46VarHandleOp*
_class
loc:@Variable_46*
_output_shapes
: *

debug_nameVariable_46/*
dtype0*
shape:�*
shared_nameVariable_46
g
,Variable_46/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_46*
_output_shapes
: 
h
Variable_46/AssignAssignVariableOpVariable_46&Variable_46/Initializer/ReadVariableOp*
dtype0
h
Variable_46/Read/ReadVariableOpReadVariableOpVariable_46*
_output_shapes	
:�*
dtype0
�
adam/dense_1_bias_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_1_bias_momentum/*
dtype0*
shape:�*+
shared_nameadam/dense_1_bias_momentum
�
.adam/dense_1_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_1_bias_momentum*
_output_shapes	
:�*
dtype0
�
&Variable_47/Initializer/ReadVariableOpReadVariableOpadam/dense_1_bias_momentum*
_class
loc:@Variable_47*
_output_shapes	
:�*
dtype0
�
Variable_47VarHandleOp*
_class
loc:@Variable_47*
_output_shapes
: *

debug_nameVariable_47/*
dtype0*
shape:�*
shared_nameVariable_47
g
,Variable_47/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_47*
_output_shapes
: 
h
Variable_47/AssignAssignVariableOpVariable_47&Variable_47/Initializer/ReadVariableOp*
dtype0
h
Variable_47/Read/ReadVariableOpReadVariableOpVariable_47*
_output_shapes	
:�*
dtype0
�
adam/dense_1_kernel_velocityVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_1_kernel_velocity/*
dtype0*
shape:
��*-
shared_nameadam/dense_1_kernel_velocity
�
0adam/dense_1_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_1_kernel_velocity* 
_output_shapes
:
��*
dtype0
�
&Variable_48/Initializer/ReadVariableOpReadVariableOpadam/dense_1_kernel_velocity*
_class
loc:@Variable_48* 
_output_shapes
:
��*
dtype0
�
Variable_48VarHandleOp*
_class
loc:@Variable_48*
_output_shapes
: *

debug_nameVariable_48/*
dtype0*
shape:
��*
shared_nameVariable_48
g
,Variable_48/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_48*
_output_shapes
: 
h
Variable_48/AssignAssignVariableOpVariable_48&Variable_48/Initializer/ReadVariableOp*
dtype0
m
Variable_48/Read/ReadVariableOpReadVariableOpVariable_48* 
_output_shapes
:
��*
dtype0
�
adam/dense_1_kernel_momentumVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_1_kernel_momentum/*
dtype0*
shape:
��*-
shared_nameadam/dense_1_kernel_momentum
�
0adam/dense_1_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_1_kernel_momentum* 
_output_shapes
:
��*
dtype0
�
&Variable_49/Initializer/ReadVariableOpReadVariableOpadam/dense_1_kernel_momentum*
_class
loc:@Variable_49* 
_output_shapes
:
��*
dtype0
�
Variable_49VarHandleOp*
_class
loc:@Variable_49*
_output_shapes
: *

debug_nameVariable_49/*
dtype0*
shape:
��*
shared_nameVariable_49
g
,Variable_49/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_49*
_output_shapes
: 
h
Variable_49/AssignAssignVariableOpVariable_49&Variable_49/Initializer/ReadVariableOp*
dtype0
m
Variable_49/Read/ReadVariableOpReadVariableOpVariable_49* 
_output_shapes
:
��*
dtype0
�
adam/dense_bias_velocityVarHandleOp*
_output_shapes
: *)

debug_nameadam/dense_bias_velocity/*
dtype0*
shape:�*)
shared_nameadam/dense_bias_velocity
�
,adam/dense_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_bias_velocity*
_output_shapes	
:�*
dtype0
�
&Variable_50/Initializer/ReadVariableOpReadVariableOpadam/dense_bias_velocity*
_class
loc:@Variable_50*
_output_shapes	
:�*
dtype0
�
Variable_50VarHandleOp*
_class
loc:@Variable_50*
_output_shapes
: *

debug_nameVariable_50/*
dtype0*
shape:�*
shared_nameVariable_50
g
,Variable_50/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_50*
_output_shapes
: 
h
Variable_50/AssignAssignVariableOpVariable_50&Variable_50/Initializer/ReadVariableOp*
dtype0
h
Variable_50/Read/ReadVariableOpReadVariableOpVariable_50*
_output_shapes	
:�*
dtype0
�
adam/dense_bias_momentumVarHandleOp*
_output_shapes
: *)

debug_nameadam/dense_bias_momentum/*
dtype0*
shape:�*)
shared_nameadam/dense_bias_momentum
�
,adam/dense_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_bias_momentum*
_output_shapes	
:�*
dtype0
�
&Variable_51/Initializer/ReadVariableOpReadVariableOpadam/dense_bias_momentum*
_class
loc:@Variable_51*
_output_shapes	
:�*
dtype0
�
Variable_51VarHandleOp*
_class
loc:@Variable_51*
_output_shapes
: *

debug_nameVariable_51/*
dtype0*
shape:�*
shared_nameVariable_51
g
,Variable_51/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_51*
_output_shapes
: 
h
Variable_51/AssignAssignVariableOpVariable_51&Variable_51/Initializer/ReadVariableOp*
dtype0
h
Variable_51/Read/ReadVariableOpReadVariableOpVariable_51*
_output_shapes	
:�*
dtype0
�
adam/dense_kernel_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_kernel_velocity/*
dtype0*
shape:	�*+
shared_nameadam/dense_kernel_velocity
�
.adam/dense_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_kernel_velocity*
_output_shapes
:	�*
dtype0
�
&Variable_52/Initializer/ReadVariableOpReadVariableOpadam/dense_kernel_velocity*
_class
loc:@Variable_52*
_output_shapes
:	�*
dtype0
�
Variable_52VarHandleOp*
_class
loc:@Variable_52*
_output_shapes
: *

debug_nameVariable_52/*
dtype0*
shape:	�*
shared_nameVariable_52
g
,Variable_52/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_52*
_output_shapes
: 
h
Variable_52/AssignAssignVariableOpVariable_52&Variable_52/Initializer/ReadVariableOp*
dtype0
l
Variable_52/Read/ReadVariableOpReadVariableOpVariable_52*
_output_shapes
:	�*
dtype0
�
adam/dense_kernel_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_kernel_momentum/*
dtype0*
shape:	�*+
shared_nameadam/dense_kernel_momentum
�
.adam/dense_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_kernel_momentum*
_output_shapes
:	�*
dtype0
�
&Variable_53/Initializer/ReadVariableOpReadVariableOpadam/dense_kernel_momentum*
_class
loc:@Variable_53*
_output_shapes
:	�*
dtype0
�
Variable_53VarHandleOp*
_class
loc:@Variable_53*
_output_shapes
: *

debug_nameVariable_53/*
dtype0*
shape:	�*
shared_nameVariable_53
g
,Variable_53/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_53*
_output_shapes
: 
h
Variable_53/AssignAssignVariableOpVariable_53&Variable_53/Initializer/ReadVariableOp*
dtype0
l
Variable_53/Read/ReadVariableOpReadVariableOpVariable_53*
_output_shapes
:	�*
dtype0
�
dense_5/biasVarHandleOp*
_output_shapes
: *

debug_namedense_5/bias/*
dtype0*
shape:d*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:d*
dtype0
�
&Variable_54/Initializer/ReadVariableOpReadVariableOpdense_5/bias*
_class
loc:@Variable_54*
_output_shapes
:d*
dtype0
�
Variable_54VarHandleOp*
_class
loc:@Variable_54*
_output_shapes
: *

debug_nameVariable_54/*
dtype0*
shape:d*
shared_nameVariable_54
g
,Variable_54/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_54*
_output_shapes
: 
h
Variable_54/AssignAssignVariableOpVariable_54&Variable_54/Initializer/ReadVariableOp*
dtype0
g
Variable_54/Read/ReadVariableOpReadVariableOpVariable_54*
_output_shapes
:d*
dtype0
�
dense_5/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_5/kernel/*
dtype0*
shape:	�d*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	�d*
dtype0
�
&Variable_55/Initializer/ReadVariableOpReadVariableOpdense_5/kernel*
_class
loc:@Variable_55*
_output_shapes
:	�d*
dtype0
�
Variable_55VarHandleOp*
_class
loc:@Variable_55*
_output_shapes
: *

debug_nameVariable_55/*
dtype0*
shape:	�d*
shared_nameVariable_55
g
,Variable_55/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_55*
_output_shapes
: 
h
Variable_55/AssignAssignVariableOpVariable_55&Variable_55/Initializer/ReadVariableOp*
dtype0
l
Variable_55/Read/ReadVariableOpReadVariableOpVariable_55*
_output_shapes
:	�d*
dtype0
�
dense_4/biasVarHandleOp*
_output_shapes
: *

debug_namedense_4/bias/*
dtype0*
shape:d*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:d*
dtype0
�
&Variable_56/Initializer/ReadVariableOpReadVariableOpdense_4/bias*
_class
loc:@Variable_56*
_output_shapes
:d*
dtype0
�
Variable_56VarHandleOp*
_class
loc:@Variable_56*
_output_shapes
: *

debug_nameVariable_56/*
dtype0*
shape:d*
shared_nameVariable_56
g
,Variable_56/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_56*
_output_shapes
: 
h
Variable_56/AssignAssignVariableOpVariable_56&Variable_56/Initializer/ReadVariableOp*
dtype0
g
Variable_56/Read/ReadVariableOpReadVariableOpVariable_56*
_output_shapes
:d*
dtype0
�
dense_4/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_4/kernel/*
dtype0*
shape:	�d*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	�d*
dtype0
�
&Variable_57/Initializer/ReadVariableOpReadVariableOpdense_4/kernel*
_class
loc:@Variable_57*
_output_shapes
:	�d*
dtype0
�
Variable_57VarHandleOp*
_class
loc:@Variable_57*
_output_shapes
: *

debug_nameVariable_57/*
dtype0*
shape:	�d*
shared_nameVariable_57
g
,Variable_57/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_57*
_output_shapes
: 
h
Variable_57/AssignAssignVariableOpVariable_57&Variable_57/Initializer/ReadVariableOp*
dtype0
l
Variable_57/Read/ReadVariableOpReadVariableOpVariable_57*
_output_shapes
:	�d*
dtype0
�
dense_3/biasVarHandleOp*
_output_shapes
: *

debug_namedense_3/bias/*
dtype0*
shape:�*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:�*
dtype0
�
&Variable_58/Initializer/ReadVariableOpReadVariableOpdense_3/bias*
_class
loc:@Variable_58*
_output_shapes	
:�*
dtype0
�
Variable_58VarHandleOp*
_class
loc:@Variable_58*
_output_shapes
: *

debug_nameVariable_58/*
dtype0*
shape:�*
shared_nameVariable_58
g
,Variable_58/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_58*
_output_shapes
: 
h
Variable_58/AssignAssignVariableOpVariable_58&Variable_58/Initializer/ReadVariableOp*
dtype0
h
Variable_58/Read/ReadVariableOpReadVariableOpVariable_58*
_output_shapes	
:�*
dtype0
�
dense_3/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_3/kernel/*
dtype0*
shape:
��*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
��*
dtype0
�
&Variable_59/Initializer/ReadVariableOpReadVariableOpdense_3/kernel*
_class
loc:@Variable_59* 
_output_shapes
:
��*
dtype0
�
Variable_59VarHandleOp*
_class
loc:@Variable_59*
_output_shapes
: *

debug_nameVariable_59/*
dtype0*
shape:
��*
shared_nameVariable_59
g
,Variable_59/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_59*
_output_shapes
: 
h
Variable_59/AssignAssignVariableOpVariable_59&Variable_59/Initializer/ReadVariableOp*
dtype0
m
Variable_59/Read/ReadVariableOpReadVariableOpVariable_59* 
_output_shapes
:
��*
dtype0
�
dense_2/biasVarHandleOp*
_output_shapes
: *

debug_namedense_2/bias/*
dtype0*
shape:�*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:�*
dtype0
�
&Variable_60/Initializer/ReadVariableOpReadVariableOpdense_2/bias*
_class
loc:@Variable_60*
_output_shapes	
:�*
dtype0
�
Variable_60VarHandleOp*
_class
loc:@Variable_60*
_output_shapes
: *

debug_nameVariable_60/*
dtype0*
shape:�*
shared_nameVariable_60
g
,Variable_60/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_60*
_output_shapes
: 
h
Variable_60/AssignAssignVariableOpVariable_60&Variable_60/Initializer/ReadVariableOp*
dtype0
h
Variable_60/Read/ReadVariableOpReadVariableOpVariable_60*
_output_shapes	
:�*
dtype0
�
dense_2/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_2/kernel/*
dtype0*
shape:
��*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
��*
dtype0
�
&Variable_61/Initializer/ReadVariableOpReadVariableOpdense_2/kernel*
_class
loc:@Variable_61* 
_output_shapes
:
��*
dtype0
�
Variable_61VarHandleOp*
_class
loc:@Variable_61*
_output_shapes
: *

debug_nameVariable_61/*
dtype0*
shape:
��*
shared_nameVariable_61
g
,Variable_61/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_61*
_output_shapes
: 
h
Variable_61/AssignAssignVariableOpVariable_61&Variable_61/Initializer/ReadVariableOp*
dtype0
m
Variable_61/Read/ReadVariableOpReadVariableOpVariable_61* 
_output_shapes
:
��*
dtype0
�
dense_1/biasVarHandleOp*
_output_shapes
: *

debug_namedense_1/bias/*
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
�
&Variable_62/Initializer/ReadVariableOpReadVariableOpdense_1/bias*
_class
loc:@Variable_62*
_output_shapes	
:�*
dtype0
�
Variable_62VarHandleOp*
_class
loc:@Variable_62*
_output_shapes
: *

debug_nameVariable_62/*
dtype0*
shape:�*
shared_nameVariable_62
g
,Variable_62/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_62*
_output_shapes
: 
h
Variable_62/AssignAssignVariableOpVariable_62&Variable_62/Initializer/ReadVariableOp*
dtype0
h
Variable_62/Read/ReadVariableOpReadVariableOpVariable_62*
_output_shapes	
:�*
dtype0
�
dense_1/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_1/kernel/*
dtype0*
shape:
��*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
��*
dtype0
�
&Variable_63/Initializer/ReadVariableOpReadVariableOpdense_1/kernel*
_class
loc:@Variable_63* 
_output_shapes
:
��*
dtype0
�
Variable_63VarHandleOp*
_class
loc:@Variable_63*
_output_shapes
: *

debug_nameVariable_63/*
dtype0*
shape:
��*
shared_nameVariable_63
g
,Variable_63/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_63*
_output_shapes
: 
h
Variable_63/AssignAssignVariableOpVariable_63&Variable_63/Initializer/ReadVariableOp*
dtype0
m
Variable_63/Read/ReadVariableOpReadVariableOpVariable_63* 
_output_shapes
:
��*
dtype0
�

dense/biasVarHandleOp*
_output_shapes
: *

debug_namedense/bias/*
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
�
&Variable_64/Initializer/ReadVariableOpReadVariableOp
dense/bias*
_class
loc:@Variable_64*
_output_shapes	
:�*
dtype0
�
Variable_64VarHandleOp*
_class
loc:@Variable_64*
_output_shapes
: *

debug_nameVariable_64/*
dtype0*
shape:�*
shared_nameVariable_64
g
,Variable_64/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_64*
_output_shapes
: 
h
Variable_64/AssignAssignVariableOpVariable_64&Variable_64/Initializer/ReadVariableOp*
dtype0
h
Variable_64/Read/ReadVariableOpReadVariableOpVariable_64*
_output_shapes	
:�*
dtype0
�
dense/kernelVarHandleOp*
_output_shapes
: *

debug_namedense/kernel/*
dtype0*
shape:	�*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�*
dtype0
�
&Variable_65/Initializer/ReadVariableOpReadVariableOpdense/kernel*
_class
loc:@Variable_65*
_output_shapes
:	�*
dtype0
�
Variable_65VarHandleOp*
_class
loc:@Variable_65*
_output_shapes
: *

debug_nameVariable_65/*
dtype0*
shape:	�*
shared_nameVariable_65
g
,Variable_65/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_65*
_output_shapes
: 
h
Variable_65/AssignAssignVariableOpVariable_65&Variable_65/Initializer/ReadVariableOp*
dtype0
l
Variable_65/Read/ReadVariableOpReadVariableOpVariable_65*
_output_shapes
:	�*
dtype0
�
adam/learning_rateVarHandleOp*
_output_shapes
: *#

debug_nameadam/learning_rate/*
dtype0*
shape: *#
shared_nameadam/learning_rate
q
&adam/learning_rate/Read/ReadVariableOpReadVariableOpadam/learning_rate*
_output_shapes
: *
dtype0
�
&Variable_66/Initializer/ReadVariableOpReadVariableOpadam/learning_rate*
_class
loc:@Variable_66*
_output_shapes
: *
dtype0
�
Variable_66VarHandleOp*
_class
loc:@Variable_66*
_output_shapes
: *

debug_nameVariable_66/*
dtype0*
shape: *
shared_nameVariable_66
g
,Variable_66/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_66*
_output_shapes
: 
h
Variable_66/AssignAssignVariableOpVariable_66&Variable_66/Initializer/ReadVariableOp*
dtype0
c
Variable_66/Read/ReadVariableOpReadVariableOpVariable_66*
_output_shapes
: *
dtype0
�
adam/iterationVarHandleOp*
_output_shapes
: *

debug_nameadam/iteration/*
dtype0	*
shape: *
shared_nameadam/iteration
i
"adam/iteration/Read/ReadVariableOpReadVariableOpadam/iteration*
_output_shapes
: *
dtype0	
�
&Variable_67/Initializer/ReadVariableOpReadVariableOpadam/iteration*
_class
loc:@Variable_67*
_output_shapes
: *
dtype0	
�
Variable_67VarHandleOp*
_class
loc:@Variable_67*
_output_shapes
: *

debug_nameVariable_67/*
dtype0	*
shape: *
shared_nameVariable_67
g
,Variable_67/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_67*
_output_shapes
: 
h
Variable_67/AssignAssignVariableOpVariable_67&Variable_67/Initializer/ReadVariableOp*
dtype0	
c
Variable_67/Read/ReadVariableOpReadVariableOpVariable_67*
_output_shapes
: *
dtype0	
y
serving_default_inputsPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *=
f8R6
4__inference_signature_wrapper_serving_default_650540

NoOpNoOp
�J
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�J
value�JB�J B�I
�
_tracked
_inbound_nodes
_outbound_nodes
_losses
_operations
_layers
_build_shapes_dict
output_names
		optimizer

_default_save_signature

signatures*
* 
* 
* 
* 
C
0
1
2
3
4
5
6
7
8*
C
0
1
2
3
4
5
6
7
8*
* 
* 
�

_variables
_trainable_variables
 _trainable_variables_indices

iterations
_learning_rate

_momentums
_velocities*

trace_0* 

serving_default* 
G
_inbound_nodes
_outbound_nodes
 _losses
!	_loss_ids* 
x
"_kernel
#bias
$_inbound_nodes
%_outbound_nodes
&_losses
'	_loss_ids
(_build_shapes_dict*
x
)_kernel
*bias
+_inbound_nodes
,_outbound_nodes
-_losses
.	_loss_ids
/_build_shapes_dict*
x
0_kernel
1bias
2_inbound_nodes
3_outbound_nodes
4_losses
5	_loss_ids
6_build_shapes_dict*
x
7_kernel
8bias
9_inbound_nodes
:_outbound_nodes
;_losses
<	_loss_ids
=_build_shapes_dict*
x
>_kernel
?bias
@_inbound_nodes
A_outbound_nodes
B_losses
C	_loss_ids
D_build_shapes_dict*
x
E_kernel
Fbias
G_inbound_nodes
H_outbound_nodes
I_losses
J	_loss_ids
K_build_shapes_dict*
n
L_inbound_nodes
M_outbound_nodes
N_losses
O	_loss_ids
P	arguments
Q_build_shapes_dict* 
�
R_tracked
S_inbound_nodes
T_outbound_nodes
U_losses
V_operations
W_layers
X_build_shapes_dict
Youtput_names
Z_default_save_signature*
�
0
1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15
i16
j17
k18
l19
m20
n21
o22
p23
q24
r25
s26
t27
u28
v29
w30
x31
y32
z33
{34
|35
}36
~37
38
�39
�40
�41
�42
�43
�44
�45*
�
"0
#1
)2
*3
04
15
76
87
>8
?9
E10
F11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21*
* 
TN
VARIABLE_VALUEVariable_67/optimizer/iterations/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEVariable_663optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
UO
VARIABLE_VALUEVariable_650_operations/1/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEVariable_64-_operations/1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
UO
VARIABLE_VALUEVariable_630_operations/2/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEVariable_62-_operations/2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
UO
VARIABLE_VALUEVariable_610_operations/3/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEVariable_60-_operations/3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
UO
VARIABLE_VALUEVariable_590_operations/4/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEVariable_58-_operations/4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
UO
VARIABLE_VALUEVariable_570_operations/5/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEVariable_56-_operations/5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
UO
VARIABLE_VALUEVariable_550_operations/6/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEVariable_54-_operations/6/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
4
�0
�1
�2
�3
�4
�5*
4
�0
�1
�2
�3
�4
�5*
* 
* 

�trace_0* 
VP
VARIABLE_VALUEVariable_531optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_521optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_511optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_501optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_491optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_481optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_471optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_461optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_452optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_442optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_432optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_422optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_412optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_402optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_392optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_382optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_372optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_362optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_352optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_342optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_332optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_322optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_312optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_302optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_292optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_282optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_272optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_262optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_252optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_242optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_232optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_222optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_212optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_202optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_192optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_182optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_172optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_162optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_152optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_142optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_132optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_122optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_112optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_102optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE
Variable_9<optimizer/_trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE
Variable_8<optimizer/_trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE
Variable_7<optimizer/_trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE
Variable_6<optimizer/_trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE
Variable_5<optimizer/_trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE
Variable_4<optimizer/_trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE
Variable_3<optimizer/_trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE
Variable_2<optimizer/_trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE
Variable_1<optimizer/_trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEVariable<optimizer/_trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
K
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids* 

�_kernel
	�bias
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids
�_build_shapes_dict*

�_kernel
	�bias
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids
�_build_shapes_dict*

�_kernel
	�bias
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids
�_build_shapes_dict*

�_kernel
	�bias
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids
�_build_shapes_dict*

�_kernel
	�bias
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids
�_build_shapes_dict*
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable_67Variable_66Variable_65Variable_64Variable_63Variable_62Variable_61Variable_60Variable_59Variable_58Variable_57Variable_56Variable_55Variable_54Variable_53Variable_52Variable_51Variable_50Variable_49Variable_48Variable_47Variable_46Variable_45Variable_44Variable_43Variable_42Variable_41Variable_40Variable_39Variable_38Variable_37Variable_36Variable_35Variable_34Variable_33Variable_32Variable_31Variable_30Variable_29Variable_28Variable_27Variable_26Variable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1VariableConst*Q
TinJ
H2F*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_651352
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable_67Variable_66Variable_65Variable_64Variable_63Variable_62Variable_61Variable_60Variable_59Variable_58Variable_57Variable_56Variable_55Variable_54Variable_53Variable_52Variable_51Variable_50Variable_49Variable_48Variable_47Variable_46Variable_45Variable_44Variable_43Variable_42Variable_41Variable_40Variable_39Variable_38Variable_37Variable_36Variable_35Variable_34Variable_33Variable_32Variable_31Variable_30Variable_29Variable_28Variable_27Variable_26Variable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_651565��
��
�&
"__inference__traced_restore_651565
file_prefix&
assignvariableop_variable_67:	 (
assignvariableop_1_variable_66: 1
assignvariableop_2_variable_65:	�-
assignvariableop_3_variable_64:	�2
assignvariableop_4_variable_63:
��-
assignvariableop_5_variable_62:	�2
assignvariableop_6_variable_61:
��-
assignvariableop_7_variable_60:	�2
assignvariableop_8_variable_59:
��-
assignvariableop_9_variable_58:	�2
assignvariableop_10_variable_57:	�d-
assignvariableop_11_variable_56:d2
assignvariableop_12_variable_55:	�d-
assignvariableop_13_variable_54:d2
assignvariableop_14_variable_53:	�2
assignvariableop_15_variable_52:	�.
assignvariableop_16_variable_51:	�.
assignvariableop_17_variable_50:	�3
assignvariableop_18_variable_49:
��3
assignvariableop_19_variable_48:
��.
assignvariableop_20_variable_47:	�.
assignvariableop_21_variable_46:	�3
assignvariableop_22_variable_45:
��3
assignvariableop_23_variable_44:
��.
assignvariableop_24_variable_43:	�.
assignvariableop_25_variable_42:	�3
assignvariableop_26_variable_41:
��3
assignvariableop_27_variable_40:
��.
assignvariableop_28_variable_39:	�.
assignvariableop_29_variable_38:	�2
assignvariableop_30_variable_37:	�d2
assignvariableop_31_variable_36:	�d-
assignvariableop_32_variable_35:d-
assignvariableop_33_variable_34:d2
assignvariableop_34_variable_33:	�d2
assignvariableop_35_variable_32:	�d-
assignvariableop_36_variable_31:d-
assignvariableop_37_variable_30:d2
assignvariableop_38_variable_29:	d�2
assignvariableop_39_variable_28:	d�.
assignvariableop_40_variable_27:	�.
assignvariableop_41_variable_26:	�3
assignvariableop_42_variable_25:
��3
assignvariableop_43_variable_24:
��.
assignvariableop_44_variable_23:	�.
assignvariableop_45_variable_22:	�3
assignvariableop_46_variable_21:
��3
assignvariableop_47_variable_20:
��.
assignvariableop_48_variable_19:	�.
assignvariableop_49_variable_18:	�3
assignvariableop_50_variable_17:
��3
assignvariableop_51_variable_16:
��.
assignvariableop_52_variable_15:	�.
assignvariableop_53_variable_14:	�2
assignvariableop_54_variable_13:	�2
assignvariableop_55_variable_12:	�-
assignvariableop_56_variable_11:-
assignvariableop_57_variable_10:1
assignvariableop_58_variable_9:	d�-
assignvariableop_59_variable_8:	�2
assignvariableop_60_variable_7:
��-
assignvariableop_61_variable_6:	�2
assignvariableop_62_variable_5:
��-
assignvariableop_63_variable_4:	�2
assignvariableop_64_variable_3:
��-
assignvariableop_65_variable_2:	�1
assignvariableop_66_variable_1:	�*
assignvariableop_67_variable:
identity_69��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*�
value�B�EB/optimizer/iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0_operations/1/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/1/bias/.ATTRIBUTES/VARIABLE_VALUEB0_operations/2/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/2/bias/.ATTRIBUTES/VARIABLE_VALUEB0_operations/3/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/3/bias/.ATTRIBUTES/VARIABLE_VALUEB0_operations/4/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/4/bias/.ATTRIBUTES/VARIABLE_VALUEB0_operations/5/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/5/bias/.ATTRIBUTES/VARIABLE_VALUEB0_operations/6/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/6/bias/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*�
value�B�EB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*S
dtypesI
G2E	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_67Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_66Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_65Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_64Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_63Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_62Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_61Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_60Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_59Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_58Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_57Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_56Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_variable_55Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_variable_54Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_variable_53Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_variable_52Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_variable_51Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_variable_50Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_variable_49Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_variable_48Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_variable_47Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_variable_46Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_variable_45Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_variable_44Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_variable_43Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_variable_42Identity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_variable_41Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_variable_40Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_variable_39Identity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_variable_38Identity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_variable_37Identity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpassignvariableop_31_variable_36Identity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_variable_35Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpassignvariableop_33_variable_34Identity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpassignvariableop_34_variable_33Identity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpassignvariableop_35_variable_32Identity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpassignvariableop_36_variable_31Identity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpassignvariableop_37_variable_30Identity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpassignvariableop_38_variable_29Identity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpassignvariableop_39_variable_28Identity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpassignvariableop_40_variable_27Identity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpassignvariableop_41_variable_26Identity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpassignvariableop_42_variable_25Identity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOpassignvariableop_43_variable_24Identity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpassignvariableop_44_variable_23Identity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpassignvariableop_45_variable_22Identity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOpassignvariableop_46_variable_21Identity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpassignvariableop_47_variable_20Identity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpassignvariableop_48_variable_19Identity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOpassignvariableop_49_variable_18Identity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpassignvariableop_50_variable_17Identity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpassignvariableop_51_variable_16Identity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpassignvariableop_52_variable_15Identity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpassignvariableop_53_variable_14Identity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpassignvariableop_54_variable_13Identity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOpassignvariableop_55_variable_12Identity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpassignvariableop_56_variable_11Identity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOpassignvariableop_57_variable_10Identity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOpassignvariableop_58_variable_9Identity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOpassignvariableop_59_variable_8Identity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOpassignvariableop_60_variable_7Identity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOpassignvariableop_61_variable_6Identity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOpassignvariableop_62_variable_5Identity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOpassignvariableop_63_variable_4Identity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOpassignvariableop_64_variable_3Identity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOpassignvariableop_65_variable_2Identity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOpassignvariableop_66_variable_1Identity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOpassignvariableop_67_variableIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_68Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_69IdentityIdentity_68:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_69Identity_69:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:(D$
"
_user_specified_name
Variable:*C&
$
_user_specified_name
Variable_1:*B&
$
_user_specified_name
Variable_2:*A&
$
_user_specified_name
Variable_3:*@&
$
_user_specified_name
Variable_4:*?&
$
_user_specified_name
Variable_5:*>&
$
_user_specified_name
Variable_6:*=&
$
_user_specified_name
Variable_7:*<&
$
_user_specified_name
Variable_8:*;&
$
_user_specified_name
Variable_9:+:'
%
_user_specified_nameVariable_10:+9'
%
_user_specified_nameVariable_11:+8'
%
_user_specified_nameVariable_12:+7'
%
_user_specified_nameVariable_13:+6'
%
_user_specified_nameVariable_14:+5'
%
_user_specified_nameVariable_15:+4'
%
_user_specified_nameVariable_16:+3'
%
_user_specified_nameVariable_17:+2'
%
_user_specified_nameVariable_18:+1'
%
_user_specified_nameVariable_19:+0'
%
_user_specified_nameVariable_20:+/'
%
_user_specified_nameVariable_21:+.'
%
_user_specified_nameVariable_22:+-'
%
_user_specified_nameVariable_23:+,'
%
_user_specified_nameVariable_24:++'
%
_user_specified_nameVariable_25:+*'
%
_user_specified_nameVariable_26:+)'
%
_user_specified_nameVariable_27:+('
%
_user_specified_nameVariable_28:+''
%
_user_specified_nameVariable_29:+&'
%
_user_specified_nameVariable_30:+%'
%
_user_specified_nameVariable_31:+$'
%
_user_specified_nameVariable_32:+#'
%
_user_specified_nameVariable_33:+"'
%
_user_specified_nameVariable_34:+!'
%
_user_specified_nameVariable_35:+ '
%
_user_specified_nameVariable_36:+'
%
_user_specified_nameVariable_37:+'
%
_user_specified_nameVariable_38:+'
%
_user_specified_nameVariable_39:+'
%
_user_specified_nameVariable_40:+'
%
_user_specified_nameVariable_41:+'
%
_user_specified_nameVariable_42:+'
%
_user_specified_nameVariable_43:+'
%
_user_specified_nameVariable_44:+'
%
_user_specified_nameVariable_45:+'
%
_user_specified_nameVariable_46:+'
%
_user_specified_nameVariable_47:+'
%
_user_specified_nameVariable_48:+'
%
_user_specified_nameVariable_49:+'
%
_user_specified_nameVariable_50:+'
%
_user_specified_nameVariable_51:+'
%
_user_specified_nameVariable_52:+'
%
_user_specified_nameVariable_53:+'
%
_user_specified_nameVariable_54:+'
%
_user_specified_nameVariable_55:+'
%
_user_specified_nameVariable_56:+'
%
_user_specified_nameVariable_57:+
'
%
_user_specified_nameVariable_58:+	'
%
_user_specified_nameVariable_59:+'
%
_user_specified_nameVariable_60:+'
%
_user_specified_nameVariable_61:+'
%
_user_specified_nameVariable_62:+'
%
_user_specified_nameVariable_63:+'
%
_user_specified_nameVariable_64:+'
%
_user_specified_nameVariable_65:+'
%
_user_specified_nameVariable_66:+'
%
_user_specified_nameVariable_67:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
4__inference_signature_wrapper_serving_default_650540

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�d
	unknown_8:d
	unknown_9:	�d

unknown_10:d

unknown_11:	d�

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:	�

unknown_20:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference_serving_default_650490o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name650536:&"
 
_user_specified_name650534:&"
 
_user_specified_name650532:&"
 
_user_specified_name650530:&"
 
_user_specified_name650528:&"
 
_user_specified_name650526:&"
 
_user_specified_name650524:&"
 
_user_specified_name650522:&"
 
_user_specified_name650520:&"
 
_user_specified_name650518:&"
 
_user_specified_name650516:&"
 
_user_specified_name650514:&
"
 
_user_specified_name650512:&	"
 
_user_specified_name650510:&"
 
_user_specified_name650508:&"
 
_user_specified_name650506:&"
 
_user_specified_name650504:&"
 
_user_specified_name650502:&"
 
_user_specified_name650500:&"
 
_user_specified_name650498:&"
 
_user_specified_name650496:&"
 
_user_specified_name650494:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�6
�
"__inference_serving_default_650626

inputsC
0decoder_1_dense_6_1_cast_readvariableop_resource:	d�>
/decoder_1_dense_6_1_add_readvariableop_resource:	�D
0decoder_1_dense_7_1_cast_readvariableop_resource:
��>
/decoder_1_dense_7_1_add_readvariableop_resource:	�D
0decoder_1_dense_8_1_cast_readvariableop_resource:
��>
/decoder_1_dense_8_1_add_readvariableop_resource:	�D
0decoder_1_dense_9_1_cast_readvariableop_resource:
��>
/decoder_1_dense_9_1_add_readvariableop_resource:	�D
1decoder_1_dense_10_1_cast_readvariableop_resource:	�>
0decoder_1_dense_10_1_add_readvariableop_resource:
identity��'decoder_1/dense_10_1/Add/ReadVariableOp�(decoder_1/dense_10_1/Cast/ReadVariableOp�&decoder_1/dense_6_1/Add/ReadVariableOp�'decoder_1/dense_6_1/Cast/ReadVariableOp�&decoder_1/dense_7_1/Add/ReadVariableOp�'decoder_1/dense_7_1/Cast/ReadVariableOp�&decoder_1/dense_8_1/Add/ReadVariableOp�'decoder_1/dense_8_1/Cast/ReadVariableOp�&decoder_1/dense_9_1/Add/ReadVariableOp�'decoder_1/dense_9_1/Cast/ReadVariableOp�
'decoder_1/dense_6_1/Cast/ReadVariableOpReadVariableOp0decoder_1_dense_6_1_cast_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
decoder_1/dense_6_1/MatMulMatMulinputs/decoder_1/dense_6_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&decoder_1/dense_6_1/Add/ReadVariableOpReadVariableOp/decoder_1_dense_6_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
decoder_1/dense_6_1/AddAddV2$decoder_1/dense_6_1/MatMul:product:0.decoder_1/dense_6_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������q
decoder_1/dense_6_1/LeakyRelu	LeakyReludecoder_1/dense_6_1/Add:z:0*(
_output_shapes
:�����������
'decoder_1/dense_7_1/Cast/ReadVariableOpReadVariableOp0decoder_1_dense_7_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
decoder_1/dense_7_1/MatMulMatMul+decoder_1/dense_6_1/LeakyRelu:activations:0/decoder_1/dense_7_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&decoder_1/dense_7_1/Add/ReadVariableOpReadVariableOp/decoder_1_dense_7_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
decoder_1/dense_7_1/AddAddV2$decoder_1/dense_7_1/MatMul:product:0.decoder_1/dense_7_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������q
decoder_1/dense_7_1/LeakyRelu	LeakyReludecoder_1/dense_7_1/Add:z:0*(
_output_shapes
:�����������
'decoder_1/dense_8_1/Cast/ReadVariableOpReadVariableOp0decoder_1_dense_8_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
decoder_1/dense_8_1/MatMulMatMul+decoder_1/dense_7_1/LeakyRelu:activations:0/decoder_1/dense_8_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&decoder_1/dense_8_1/Add/ReadVariableOpReadVariableOp/decoder_1_dense_8_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
decoder_1/dense_8_1/AddAddV2$decoder_1/dense_8_1/MatMul:product:0.decoder_1/dense_8_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������q
decoder_1/dense_8_1/LeakyRelu	LeakyReludecoder_1/dense_8_1/Add:z:0*(
_output_shapes
:�����������
'decoder_1/dense_9_1/Cast/ReadVariableOpReadVariableOp0decoder_1_dense_9_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
decoder_1/dense_9_1/MatMulMatMul+decoder_1/dense_8_1/LeakyRelu:activations:0/decoder_1/dense_9_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&decoder_1/dense_9_1/Add/ReadVariableOpReadVariableOp/decoder_1_dense_9_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
decoder_1/dense_9_1/AddAddV2$decoder_1/dense_9_1/MatMul:product:0.decoder_1/dense_9_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������q
decoder_1/dense_9_1/LeakyRelu	LeakyReludecoder_1/dense_9_1/Add:z:0*(
_output_shapes
:�����������
(decoder_1/dense_10_1/Cast/ReadVariableOpReadVariableOp1decoder_1_dense_10_1_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
decoder_1/dense_10_1/MatMulMatMul+decoder_1/dense_9_1/LeakyRelu:activations:00decoder_1/dense_10_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'decoder_1/dense_10_1/Add/ReadVariableOpReadVariableOp0decoder_1_dense_10_1_add_readvariableop_resource*
_output_shapes
:*
dtype0�
decoder_1/dense_10_1/AddAddV2%decoder_1/dense_10_1/MatMul:product:0/decoder_1/dense_10_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������k
IdentityIdentitydecoder_1/dense_10_1/Add:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^decoder_1/dense_10_1/Add/ReadVariableOp)^decoder_1/dense_10_1/Cast/ReadVariableOp'^decoder_1/dense_6_1/Add/ReadVariableOp(^decoder_1/dense_6_1/Cast/ReadVariableOp'^decoder_1/dense_7_1/Add/ReadVariableOp(^decoder_1/dense_7_1/Cast/ReadVariableOp'^decoder_1/dense_8_1/Add/ReadVariableOp(^decoder_1/dense_8_1/Cast/ReadVariableOp'^decoder_1/dense_9_1/Add/ReadVariableOp(^decoder_1/dense_9_1/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������d: : : : : : : : : : 2R
'decoder_1/dense_10_1/Add/ReadVariableOp'decoder_1/dense_10_1/Add/ReadVariableOp2T
(decoder_1/dense_10_1/Cast/ReadVariableOp(decoder_1/dense_10_1/Cast/ReadVariableOp2P
&decoder_1/dense_6_1/Add/ReadVariableOp&decoder_1/dense_6_1/Add/ReadVariableOp2R
'decoder_1/dense_6_1/Cast/ReadVariableOp'decoder_1/dense_6_1/Cast/ReadVariableOp2P
&decoder_1/dense_7_1/Add/ReadVariableOp&decoder_1/dense_7_1/Add/ReadVariableOp2R
'decoder_1/dense_7_1/Cast/ReadVariableOp'decoder_1/dense_7_1/Cast/ReadVariableOp2P
&decoder_1/dense_8_1/Add/ReadVariableOp&decoder_1/dense_8_1/Add/ReadVariableOp2R
'decoder_1/dense_8_1/Cast/ReadVariableOp'decoder_1/dense_8_1/Cast/ReadVariableOp2P
&decoder_1/dense_9_1/Add/ReadVariableOp&decoder_1/dense_9_1/Add/ReadVariableOp2R
'decoder_1/dense_9_1/Cast/ReadVariableOp'decoder_1/dense_9_1/Cast/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
��
�
"__inference_serving_default_650490

inputs=
*vae_1_dense_1_cast_readvariableop_resource:	�8
)vae_1_dense_1_add_readvariableop_resource:	�@
,vae_1_dense_1_2_cast_readvariableop_resource:
��:
+vae_1_dense_1_2_add_readvariableop_resource:	�@
,vae_1_dense_2_1_cast_readvariableop_resource:
��:
+vae_1_dense_2_1_add_readvariableop_resource:	�@
,vae_1_dense_3_1_cast_readvariableop_resource:
��:
+vae_1_dense_3_1_add_readvariableop_resource:	�?
,vae_1_dense_4_1_cast_readvariableop_resource:	�d9
+vae_1_dense_4_1_add_readvariableop_resource:d?
,vae_1_dense_5_1_cast_readvariableop_resource:	�d9
+vae_1_dense_5_1_add_readvariableop_resource:dI
6vae_1_decoder_1_dense_6_1_cast_readvariableop_resource:	d�D
5vae_1_decoder_1_dense_6_1_add_readvariableop_resource:	�J
6vae_1_decoder_1_dense_7_1_cast_readvariableop_resource:
��D
5vae_1_decoder_1_dense_7_1_add_readvariableop_resource:	�J
6vae_1_decoder_1_dense_8_1_cast_readvariableop_resource:
��D
5vae_1_decoder_1_dense_8_1_add_readvariableop_resource:	�J
6vae_1_decoder_1_dense_9_1_cast_readvariableop_resource:
��D
5vae_1_decoder_1_dense_9_1_add_readvariableop_resource:	�J
7vae_1_decoder_1_dense_10_1_cast_readvariableop_resource:	�D
6vae_1_decoder_1_dense_10_1_add_readvariableop_resource:
identity��-vae_1/decoder_1/dense_10_1/Add/ReadVariableOp�.vae_1/decoder_1/dense_10_1/Cast/ReadVariableOp�,vae_1/decoder_1/dense_6_1/Add/ReadVariableOp�-vae_1/decoder_1/dense_6_1/Cast/ReadVariableOp�,vae_1/decoder_1/dense_7_1/Add/ReadVariableOp�-vae_1/decoder_1/dense_7_1/Cast/ReadVariableOp�,vae_1/decoder_1/dense_8_1/Add/ReadVariableOp�-vae_1/decoder_1/dense_8_1/Cast/ReadVariableOp�,vae_1/decoder_1/dense_9_1/Add/ReadVariableOp�-vae_1/decoder_1/dense_9_1/Cast/ReadVariableOp� vae_1/dense_1/Add/ReadVariableOp�!vae_1/dense_1/Cast/ReadVariableOp�"vae_1/dense_1_2/Add/ReadVariableOp�#vae_1/dense_1_2/Cast/ReadVariableOp�"vae_1/dense_2_1/Add/ReadVariableOp�#vae_1/dense_2_1/Cast/ReadVariableOp�"vae_1/dense_3_1/Add/ReadVariableOp�#vae_1/dense_3_1/Cast/ReadVariableOp�"vae_1/dense_4_1/Add/ReadVariableOp�#vae_1/dense_4_1/Cast/ReadVariableOp�"vae_1/dense_5_1/Add/ReadVariableOp�#vae_1/dense_5_1/Cast/ReadVariableOp�
!vae_1/dense_1/Cast/ReadVariableOpReadVariableOp*vae_1_dense_1_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
vae_1/dense_1/MatMulMatMulinputs)vae_1/dense_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 vae_1/dense_1/Add/ReadVariableOpReadVariableOp)vae_1_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
vae_1/dense_1/AddAddV2vae_1/dense_1/MatMul:product:0(vae_1/dense_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
vae_1/dense_1/LeakyRelu	LeakyReluvae_1/dense_1/Add:z:0*(
_output_shapes
:�����������
#vae_1/dense_1_2/Cast/ReadVariableOpReadVariableOp,vae_1_dense_1_2_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
vae_1/dense_1_2/MatMulMatMul%vae_1/dense_1/LeakyRelu:activations:0+vae_1/dense_1_2/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"vae_1/dense_1_2/Add/ReadVariableOpReadVariableOp+vae_1_dense_1_2_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
vae_1/dense_1_2/AddAddV2 vae_1/dense_1_2/MatMul:product:0*vae_1/dense_1_2/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
vae_1/dense_1_2/LeakyRelu	LeakyReluvae_1/dense_1_2/Add:z:0*(
_output_shapes
:�����������
#vae_1/dense_2_1/Cast/ReadVariableOpReadVariableOp,vae_1_dense_2_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
vae_1/dense_2_1/MatMulMatMul'vae_1/dense_1_2/LeakyRelu:activations:0+vae_1/dense_2_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"vae_1/dense_2_1/Add/ReadVariableOpReadVariableOp+vae_1_dense_2_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
vae_1/dense_2_1/AddAddV2 vae_1/dense_2_1/MatMul:product:0*vae_1/dense_2_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
vae_1/dense_2_1/LeakyRelu	LeakyReluvae_1/dense_2_1/Add:z:0*(
_output_shapes
:�����������
#vae_1/dense_3_1/Cast/ReadVariableOpReadVariableOp,vae_1_dense_3_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
vae_1/dense_3_1/MatMulMatMul'vae_1/dense_2_1/LeakyRelu:activations:0+vae_1/dense_3_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"vae_1/dense_3_1/Add/ReadVariableOpReadVariableOp+vae_1_dense_3_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
vae_1/dense_3_1/AddAddV2 vae_1/dense_3_1/MatMul:product:0*vae_1/dense_3_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
vae_1/dense_3_1/LeakyRelu	LeakyReluvae_1/dense_3_1/Add:z:0*(
_output_shapes
:�����������
#vae_1/dense_4_1/Cast/ReadVariableOpReadVariableOp,vae_1_dense_4_1_cast_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
vae_1/dense_4_1/MatMulMatMul'vae_1/dense_3_1/LeakyRelu:activations:0+vae_1/dense_4_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
"vae_1/dense_4_1/Add/ReadVariableOpReadVariableOp+vae_1_dense_4_1_add_readvariableop_resource*
_output_shapes
:d*
dtype0�
vae_1/dense_4_1/AddAddV2 vae_1/dense_4_1/MatMul:product:0*vae_1/dense_4_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
#vae_1/dense_5_1/Cast/ReadVariableOpReadVariableOp,vae_1_dense_5_1_cast_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
vae_1/dense_5_1/MatMulMatMul'vae_1/dense_3_1/LeakyRelu:activations:0+vae_1/dense_5_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
"vae_1/dense_5_1/Add/ReadVariableOpReadVariableOp+vae_1_dense_5_1_add_readvariableop_resource*
_output_shapes
:d*
dtype0�
vae_1/dense_5_1/AddAddV2 vae_1/dense_5_1/MatMul:product:0*vae_1/dense_5_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dd
vae_1/z_1/ShapeShapevae_1/dense_4_1/Add:z:0*
T0*
_output_shapes
::��g
vae_1/z_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
vae_1/z_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
vae_1/z_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
vae_1/z_1/strided_sliceStridedSlicevae_1/z_1/Shape:output:0&vae_1/z_1/strided_slice/stack:output:0(vae_1/z_1/strided_slice/stack_1:output:0(vae_1/z_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
vae_1/z_1/Shape_1Shapevae_1/dense_4_1/Add:z:0*
T0*
_output_shapes
::��i
vae_1/z_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:k
!vae_1/z_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!vae_1/z_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
vae_1/z_1/strided_slice_1StridedSlicevae_1/z_1/Shape_1:output:0(vae_1/z_1/strided_slice_1/stack:output:0*vae_1/z_1/strided_slice_1/stack_1:output:0*vae_1/z_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
vae_1/z_1/random_normal/shapePack vae_1/z_1/strided_slice:output:0"vae_1/z_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:a
vae_1/z_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    c
vae_1/z_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,vae_1/z_1/random_normal/RandomStandardNormalRandomStandardNormal&vae_1/z_1/random_normal/shape:output:0*
T0*'
_output_shapes
:���������d*
dtype0*
seed2��)*
seed���)�
vae_1/z_1/random_normal/mulMul5vae_1/z_1/random_normal/RandomStandardNormal:output:0'vae_1/z_1/random_normal/stddev:output:0*
T0*'
_output_shapes
:���������d�
vae_1/z_1/random_normalAddV2vae_1/z_1/random_normal/mul:z:0%vae_1/z_1/random_normal/mean:output:0*
T0*'
_output_shapes
:���������dT
vae_1/z_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?y
vae_1/z_1/mulMulvae_1/z_1/mul/x:output:0vae_1/dense_5_1/Add:z:0*
T0*'
_output_shapes
:���������dY
vae_1/z_1/ExpExpvae_1/z_1/mul:z:0*
T0*'
_output_shapes
:���������dx
vae_1/z_1/mul_1Mulvae_1/z_1/Exp:y:0vae_1/z_1/random_normal:z:0*
T0*'
_output_shapes
:���������dv
vae_1/z_1/addAddV2vae_1/dense_4_1/Add:z:0vae_1/z_1/mul_1:z:0*
T0*'
_output_shapes
:���������d�
-vae_1/decoder_1/dense_6_1/Cast/ReadVariableOpReadVariableOp6vae_1_decoder_1_dense_6_1_cast_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
 vae_1/decoder_1/dense_6_1/MatMulMatMulvae_1/z_1/add:z:05vae_1/decoder_1/dense_6_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,vae_1/decoder_1/dense_6_1/Add/ReadVariableOpReadVariableOp5vae_1_decoder_1_dense_6_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
vae_1/decoder_1/dense_6_1/AddAddV2*vae_1/decoder_1/dense_6_1/MatMul:product:04vae_1/decoder_1/dense_6_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
#vae_1/decoder_1/dense_6_1/LeakyRelu	LeakyRelu!vae_1/decoder_1/dense_6_1/Add:z:0*(
_output_shapes
:�����������
-vae_1/decoder_1/dense_7_1/Cast/ReadVariableOpReadVariableOp6vae_1_decoder_1_dense_7_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
 vae_1/decoder_1/dense_7_1/MatMulMatMul1vae_1/decoder_1/dense_6_1/LeakyRelu:activations:05vae_1/decoder_1/dense_7_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,vae_1/decoder_1/dense_7_1/Add/ReadVariableOpReadVariableOp5vae_1_decoder_1_dense_7_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
vae_1/decoder_1/dense_7_1/AddAddV2*vae_1/decoder_1/dense_7_1/MatMul:product:04vae_1/decoder_1/dense_7_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
#vae_1/decoder_1/dense_7_1/LeakyRelu	LeakyRelu!vae_1/decoder_1/dense_7_1/Add:z:0*(
_output_shapes
:�����������
-vae_1/decoder_1/dense_8_1/Cast/ReadVariableOpReadVariableOp6vae_1_decoder_1_dense_8_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
 vae_1/decoder_1/dense_8_1/MatMulMatMul1vae_1/decoder_1/dense_7_1/LeakyRelu:activations:05vae_1/decoder_1/dense_8_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,vae_1/decoder_1/dense_8_1/Add/ReadVariableOpReadVariableOp5vae_1_decoder_1_dense_8_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
vae_1/decoder_1/dense_8_1/AddAddV2*vae_1/decoder_1/dense_8_1/MatMul:product:04vae_1/decoder_1/dense_8_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
#vae_1/decoder_1/dense_8_1/LeakyRelu	LeakyRelu!vae_1/decoder_1/dense_8_1/Add:z:0*(
_output_shapes
:�����������
-vae_1/decoder_1/dense_9_1/Cast/ReadVariableOpReadVariableOp6vae_1_decoder_1_dense_9_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
 vae_1/decoder_1/dense_9_1/MatMulMatMul1vae_1/decoder_1/dense_8_1/LeakyRelu:activations:05vae_1/decoder_1/dense_9_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,vae_1/decoder_1/dense_9_1/Add/ReadVariableOpReadVariableOp5vae_1_decoder_1_dense_9_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
vae_1/decoder_1/dense_9_1/AddAddV2*vae_1/decoder_1/dense_9_1/MatMul:product:04vae_1/decoder_1/dense_9_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
#vae_1/decoder_1/dense_9_1/LeakyRelu	LeakyRelu!vae_1/decoder_1/dense_9_1/Add:z:0*(
_output_shapes
:�����������
.vae_1/decoder_1/dense_10_1/Cast/ReadVariableOpReadVariableOp7vae_1_decoder_1_dense_10_1_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
!vae_1/decoder_1/dense_10_1/MatMulMatMul1vae_1/decoder_1/dense_9_1/LeakyRelu:activations:06vae_1/decoder_1/dense_10_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-vae_1/decoder_1/dense_10_1/Add/ReadVariableOpReadVariableOp6vae_1_decoder_1_dense_10_1_add_readvariableop_resource*
_output_shapes
:*
dtype0�
vae_1/decoder_1/dense_10_1/AddAddV2+vae_1/decoder_1/dense_10_1/MatMul:product:05vae_1/decoder_1/dense_10_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
IdentityIdentity"vae_1/decoder_1/dense_10_1/Add:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^vae_1/decoder_1/dense_10_1/Add/ReadVariableOp/^vae_1/decoder_1/dense_10_1/Cast/ReadVariableOp-^vae_1/decoder_1/dense_6_1/Add/ReadVariableOp.^vae_1/decoder_1/dense_6_1/Cast/ReadVariableOp-^vae_1/decoder_1/dense_7_1/Add/ReadVariableOp.^vae_1/decoder_1/dense_7_1/Cast/ReadVariableOp-^vae_1/decoder_1/dense_8_1/Add/ReadVariableOp.^vae_1/decoder_1/dense_8_1/Cast/ReadVariableOp-^vae_1/decoder_1/dense_9_1/Add/ReadVariableOp.^vae_1/decoder_1/dense_9_1/Cast/ReadVariableOp!^vae_1/dense_1/Add/ReadVariableOp"^vae_1/dense_1/Cast/ReadVariableOp#^vae_1/dense_1_2/Add/ReadVariableOp$^vae_1/dense_1_2/Cast/ReadVariableOp#^vae_1/dense_2_1/Add/ReadVariableOp$^vae_1/dense_2_1/Cast/ReadVariableOp#^vae_1/dense_3_1/Add/ReadVariableOp$^vae_1/dense_3_1/Cast/ReadVariableOp#^vae_1/dense_4_1/Add/ReadVariableOp$^vae_1/dense_4_1/Cast/ReadVariableOp#^vae_1/dense_5_1/Add/ReadVariableOp$^vae_1/dense_5_1/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : : : : : 2^
-vae_1/decoder_1/dense_10_1/Add/ReadVariableOp-vae_1/decoder_1/dense_10_1/Add/ReadVariableOp2`
.vae_1/decoder_1/dense_10_1/Cast/ReadVariableOp.vae_1/decoder_1/dense_10_1/Cast/ReadVariableOp2\
,vae_1/decoder_1/dense_6_1/Add/ReadVariableOp,vae_1/decoder_1/dense_6_1/Add/ReadVariableOp2^
-vae_1/decoder_1/dense_6_1/Cast/ReadVariableOp-vae_1/decoder_1/dense_6_1/Cast/ReadVariableOp2\
,vae_1/decoder_1/dense_7_1/Add/ReadVariableOp,vae_1/decoder_1/dense_7_1/Add/ReadVariableOp2^
-vae_1/decoder_1/dense_7_1/Cast/ReadVariableOp-vae_1/decoder_1/dense_7_1/Cast/ReadVariableOp2\
,vae_1/decoder_1/dense_8_1/Add/ReadVariableOp,vae_1/decoder_1/dense_8_1/Add/ReadVariableOp2^
-vae_1/decoder_1/dense_8_1/Cast/ReadVariableOp-vae_1/decoder_1/dense_8_1/Cast/ReadVariableOp2\
,vae_1/decoder_1/dense_9_1/Add/ReadVariableOp,vae_1/decoder_1/dense_9_1/Add/ReadVariableOp2^
-vae_1/decoder_1/dense_9_1/Cast/ReadVariableOp-vae_1/decoder_1/dense_9_1/Cast/ReadVariableOp2D
 vae_1/dense_1/Add/ReadVariableOp vae_1/dense_1/Add/ReadVariableOp2F
!vae_1/dense_1/Cast/ReadVariableOp!vae_1/dense_1/Cast/ReadVariableOp2H
"vae_1/dense_1_2/Add/ReadVariableOp"vae_1/dense_1_2/Add/ReadVariableOp2J
#vae_1/dense_1_2/Cast/ReadVariableOp#vae_1/dense_1_2/Cast/ReadVariableOp2H
"vae_1/dense_2_1/Add/ReadVariableOp"vae_1/dense_2_1/Add/ReadVariableOp2J
#vae_1/dense_2_1/Cast/ReadVariableOp#vae_1/dense_2_1/Cast/ReadVariableOp2H
"vae_1/dense_3_1/Add/ReadVariableOp"vae_1/dense_3_1/Add/ReadVariableOp2J
#vae_1/dense_3_1/Cast/ReadVariableOp#vae_1/dense_3_1/Cast/ReadVariableOp2H
"vae_1/dense_4_1/Add/ReadVariableOp"vae_1/dense_4_1/Add/ReadVariableOp2J
#vae_1/dense_4_1/Cast/ReadVariableOp#vae_1/dense_4_1/Cast/ReadVariableOp2H
"vae_1/dense_5_1/Add/ReadVariableOp"vae_1/dense_5_1/Add/ReadVariableOp2J
#vae_1/dense_5_1/Cast/ReadVariableOp#vae_1/dense_5_1/Cast/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�:
__inference__traced_save_651352
file_prefix,
"read_disablecopyonread_variable_67:	 .
$read_1_disablecopyonread_variable_66: 7
$read_2_disablecopyonread_variable_65:	�3
$read_3_disablecopyonread_variable_64:	�8
$read_4_disablecopyonread_variable_63:
��3
$read_5_disablecopyonread_variable_62:	�8
$read_6_disablecopyonread_variable_61:
��3
$read_7_disablecopyonread_variable_60:	�8
$read_8_disablecopyonread_variable_59:
��3
$read_9_disablecopyonread_variable_58:	�8
%read_10_disablecopyonread_variable_57:	�d3
%read_11_disablecopyonread_variable_56:d8
%read_12_disablecopyonread_variable_55:	�d3
%read_13_disablecopyonread_variable_54:d8
%read_14_disablecopyonread_variable_53:	�8
%read_15_disablecopyonread_variable_52:	�4
%read_16_disablecopyonread_variable_51:	�4
%read_17_disablecopyonread_variable_50:	�9
%read_18_disablecopyonread_variable_49:
��9
%read_19_disablecopyonread_variable_48:
��4
%read_20_disablecopyonread_variable_47:	�4
%read_21_disablecopyonread_variable_46:	�9
%read_22_disablecopyonread_variable_45:
��9
%read_23_disablecopyonread_variable_44:
��4
%read_24_disablecopyonread_variable_43:	�4
%read_25_disablecopyonread_variable_42:	�9
%read_26_disablecopyonread_variable_41:
��9
%read_27_disablecopyonread_variable_40:
��4
%read_28_disablecopyonread_variable_39:	�4
%read_29_disablecopyonread_variable_38:	�8
%read_30_disablecopyonread_variable_37:	�d8
%read_31_disablecopyonread_variable_36:	�d3
%read_32_disablecopyonread_variable_35:d3
%read_33_disablecopyonread_variable_34:d8
%read_34_disablecopyonread_variable_33:	�d8
%read_35_disablecopyonread_variable_32:	�d3
%read_36_disablecopyonread_variable_31:d3
%read_37_disablecopyonread_variable_30:d8
%read_38_disablecopyonread_variable_29:	d�8
%read_39_disablecopyonread_variable_28:	d�4
%read_40_disablecopyonread_variable_27:	�4
%read_41_disablecopyonread_variable_26:	�9
%read_42_disablecopyonread_variable_25:
��9
%read_43_disablecopyonread_variable_24:
��4
%read_44_disablecopyonread_variable_23:	�4
%read_45_disablecopyonread_variable_22:	�9
%read_46_disablecopyonread_variable_21:
��9
%read_47_disablecopyonread_variable_20:
��4
%read_48_disablecopyonread_variable_19:	�4
%read_49_disablecopyonread_variable_18:	�9
%read_50_disablecopyonread_variable_17:
��9
%read_51_disablecopyonread_variable_16:
��4
%read_52_disablecopyonread_variable_15:	�4
%read_53_disablecopyonread_variable_14:	�8
%read_54_disablecopyonread_variable_13:	�8
%read_55_disablecopyonread_variable_12:	�3
%read_56_disablecopyonread_variable_11:3
%read_57_disablecopyonread_variable_10:7
$read_58_disablecopyonread_variable_9:	d�3
$read_59_disablecopyonread_variable_8:	�8
$read_60_disablecopyonread_variable_7:
��3
$read_61_disablecopyonread_variable_6:	�8
$read_62_disablecopyonread_variable_5:
��3
$read_63_disablecopyonread_variable_4:	�8
$read_64_disablecopyonread_variable_3:
��3
$read_65_disablecopyonread_variable_2:	�7
$read_66_disablecopyonread_variable_1:	�0
"read_67_disablecopyonread_variable:
savev2_const
identity_137��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
: e
Read/DisableCopyOnReadDisableCopyOnRead"read_disablecopyonread_variable_67*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp"read_disablecopyonread_variable_67^Read/DisableCopyOnRead*
_output_shapes
: *
dtype0	R
IdentityIdentityRead/ReadVariableOp:value:0*
T0	*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0	*
_output_shapes
: i
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_variable_66*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_variable_66^Read_1/DisableCopyOnRead*
_output_shapes
: *
dtype0V

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
: [

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_2/DisableCopyOnReadDisableCopyOnRead$read_2_disablecopyonread_variable_65*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp$read_2_disablecopyonread_variable_65^Read_2/DisableCopyOnRead*
_output_shapes
:	�*
dtype0_

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	�i
Read_3/DisableCopyOnReadDisableCopyOnRead$read_3_disablecopyonread_variable_64*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp$read_3_disablecopyonread_variable_64^Read_3/DisableCopyOnRead*
_output_shapes	
:�*
dtype0[

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�i
Read_4/DisableCopyOnReadDisableCopyOnRead$read_4_disablecopyonread_variable_63*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp$read_4_disablecopyonread_variable_63^Read_4/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0`

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_variable_62*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp$read_5_disablecopyonread_variable_62^Read_5/DisableCopyOnRead*
_output_shapes	
:�*
dtype0\
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:�i
Read_6/DisableCopyOnReadDisableCopyOnRead$read_6_disablecopyonread_variable_61*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp$read_6_disablecopyonread_variable_61^Read_6/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0a
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_variable_60*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_variable_60^Read_7/DisableCopyOnRead*
_output_shapes	
:�*
dtype0\
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:�i
Read_8/DisableCopyOnReadDisableCopyOnRead$read_8_disablecopyonread_variable_59*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp$read_8_disablecopyonread_variable_59^Read_8/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0a
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Read_9/DisableCopyOnReadDisableCopyOnRead$read_9_disablecopyonread_variable_58*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp$read_9_disablecopyonread_variable_58^Read_9/DisableCopyOnRead*
_output_shapes	
:�*
dtype0\
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_10/DisableCopyOnReadDisableCopyOnRead%read_10_disablecopyonread_variable_57*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp%read_10_disablecopyonread_variable_57^Read_10/DisableCopyOnRead*
_output_shapes
:	�d*
dtype0a
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*
_output_shapes
:	�df
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	�dk
Read_11/DisableCopyOnReadDisableCopyOnRead%read_11_disablecopyonread_variable_56*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp%read_11_disablecopyonread_variable_56^Read_11/DisableCopyOnRead*
_output_shapes
:d*
dtype0\
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*
_output_shapes
:da
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:dk
Read_12/DisableCopyOnReadDisableCopyOnRead%read_12_disablecopyonread_variable_55*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp%read_12_disablecopyonread_variable_55^Read_12/DisableCopyOnRead*
_output_shapes
:	�d*
dtype0a
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*
_output_shapes
:	�df
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	�dk
Read_13/DisableCopyOnReadDisableCopyOnRead%read_13_disablecopyonread_variable_54*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp%read_13_disablecopyonread_variable_54^Read_13/DisableCopyOnRead*
_output_shapes
:d*
dtype0\
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0*
_output_shapes
:da
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:dk
Read_14/DisableCopyOnReadDisableCopyOnRead%read_14_disablecopyonread_variable_53*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp%read_14_disablecopyonread_variable_53^Read_14/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_28IdentityRead_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:	�k
Read_15/DisableCopyOnReadDisableCopyOnRead%read_15_disablecopyonread_variable_52*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp%read_15_disablecopyonread_variable_52^Read_15/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_30IdentityRead_15/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:	�k
Read_16/DisableCopyOnReadDisableCopyOnRead%read_16_disablecopyonread_variable_51*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp%read_16_disablecopyonread_variable_51^Read_16/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_32IdentityRead_16/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_17/DisableCopyOnReadDisableCopyOnRead%read_17_disablecopyonread_variable_50*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp%read_17_disablecopyonread_variable_50^Read_17/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_34IdentityRead_17/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_18/DisableCopyOnReadDisableCopyOnRead%read_18_disablecopyonread_variable_49*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp%read_18_disablecopyonread_variable_49^Read_18/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_36IdentityRead_18/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_19/DisableCopyOnReadDisableCopyOnRead%read_19_disablecopyonread_variable_48*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp%read_19_disablecopyonread_variable_48^Read_19/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_38IdentityRead_19/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_20/DisableCopyOnReadDisableCopyOnRead%read_20_disablecopyonread_variable_47*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp%read_20_disablecopyonread_variable_47^Read_20/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_40IdentityRead_20/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_21/DisableCopyOnReadDisableCopyOnRead%read_21_disablecopyonread_variable_46*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp%read_21_disablecopyonread_variable_46^Read_21/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_42IdentityRead_21/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_22/DisableCopyOnReadDisableCopyOnRead%read_22_disablecopyonread_variable_45*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp%read_22_disablecopyonread_variable_45^Read_22/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_44IdentityRead_22/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_23/DisableCopyOnReadDisableCopyOnRead%read_23_disablecopyonread_variable_44*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp%read_23_disablecopyonread_variable_44^Read_23/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_46IdentityRead_23/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_24/DisableCopyOnReadDisableCopyOnRead%read_24_disablecopyonread_variable_43*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp%read_24_disablecopyonread_variable_43^Read_24/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_48IdentityRead_24/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_25/DisableCopyOnReadDisableCopyOnRead%read_25_disablecopyonread_variable_42*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp%read_25_disablecopyonread_variable_42^Read_25/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_50IdentityRead_25/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_26/DisableCopyOnReadDisableCopyOnRead%read_26_disablecopyonread_variable_41*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp%read_26_disablecopyonread_variable_41^Read_26/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_52IdentityRead_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_27/DisableCopyOnReadDisableCopyOnRead%read_27_disablecopyonread_variable_40*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp%read_27_disablecopyonread_variable_40^Read_27/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_54IdentityRead_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_28/DisableCopyOnReadDisableCopyOnRead%read_28_disablecopyonread_variable_39*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp%read_28_disablecopyonread_variable_39^Read_28/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_56IdentityRead_28/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_29/DisableCopyOnReadDisableCopyOnRead%read_29_disablecopyonread_variable_38*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp%read_29_disablecopyonread_variable_38^Read_29/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_58IdentityRead_29/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_30/DisableCopyOnReadDisableCopyOnRead%read_30_disablecopyonread_variable_37*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp%read_30_disablecopyonread_variable_37^Read_30/DisableCopyOnRead*
_output_shapes
:	�d*
dtype0a
Identity_60IdentityRead_30/ReadVariableOp:value:0*
T0*
_output_shapes
:	�df
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:	�dk
Read_31/DisableCopyOnReadDisableCopyOnRead%read_31_disablecopyonread_variable_36*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp%read_31_disablecopyonread_variable_36^Read_31/DisableCopyOnRead*
_output_shapes
:	�d*
dtype0a
Identity_62IdentityRead_31/ReadVariableOp:value:0*
T0*
_output_shapes
:	�df
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:	�dk
Read_32/DisableCopyOnReadDisableCopyOnRead%read_32_disablecopyonread_variable_35*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp%read_32_disablecopyonread_variable_35^Read_32/DisableCopyOnRead*
_output_shapes
:d*
dtype0\
Identity_64IdentityRead_32/ReadVariableOp:value:0*
T0*
_output_shapes
:da
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:dk
Read_33/DisableCopyOnReadDisableCopyOnRead%read_33_disablecopyonread_variable_34*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp%read_33_disablecopyonread_variable_34^Read_33/DisableCopyOnRead*
_output_shapes
:d*
dtype0\
Identity_66IdentityRead_33/ReadVariableOp:value:0*
T0*
_output_shapes
:da
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:dk
Read_34/DisableCopyOnReadDisableCopyOnRead%read_34_disablecopyonread_variable_33*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp%read_34_disablecopyonread_variable_33^Read_34/DisableCopyOnRead*
_output_shapes
:	�d*
dtype0a
Identity_68IdentityRead_34/ReadVariableOp:value:0*
T0*
_output_shapes
:	�df
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:	�dk
Read_35/DisableCopyOnReadDisableCopyOnRead%read_35_disablecopyonread_variable_32*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp%read_35_disablecopyonread_variable_32^Read_35/DisableCopyOnRead*
_output_shapes
:	�d*
dtype0a
Identity_70IdentityRead_35/ReadVariableOp:value:0*
T0*
_output_shapes
:	�df
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:	�dk
Read_36/DisableCopyOnReadDisableCopyOnRead%read_36_disablecopyonread_variable_31*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp%read_36_disablecopyonread_variable_31^Read_36/DisableCopyOnRead*
_output_shapes
:d*
dtype0\
Identity_72IdentityRead_36/ReadVariableOp:value:0*
T0*
_output_shapes
:da
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:dk
Read_37/DisableCopyOnReadDisableCopyOnRead%read_37_disablecopyonread_variable_30*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp%read_37_disablecopyonread_variable_30^Read_37/DisableCopyOnRead*
_output_shapes
:d*
dtype0\
Identity_74IdentityRead_37/ReadVariableOp:value:0*
T0*
_output_shapes
:da
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:dk
Read_38/DisableCopyOnReadDisableCopyOnRead%read_38_disablecopyonread_variable_29*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp%read_38_disablecopyonread_variable_29^Read_38/DisableCopyOnRead*
_output_shapes
:	d�*
dtype0a
Identity_76IdentityRead_38/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�f
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:	d�k
Read_39/DisableCopyOnReadDisableCopyOnRead%read_39_disablecopyonread_variable_28*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp%read_39_disablecopyonread_variable_28^Read_39/DisableCopyOnRead*
_output_shapes
:	d�*
dtype0a
Identity_78IdentityRead_39/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�f
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:	d�k
Read_40/DisableCopyOnReadDisableCopyOnRead%read_40_disablecopyonread_variable_27*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp%read_40_disablecopyonread_variable_27^Read_40/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_80IdentityRead_40/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_41/DisableCopyOnReadDisableCopyOnRead%read_41_disablecopyonread_variable_26*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp%read_41_disablecopyonread_variable_26^Read_41/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_82IdentityRead_41/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_42/DisableCopyOnReadDisableCopyOnRead%read_42_disablecopyonread_variable_25*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp%read_42_disablecopyonread_variable_25^Read_42/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_84IdentityRead_42/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_43/DisableCopyOnReadDisableCopyOnRead%read_43_disablecopyonread_variable_24*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp%read_43_disablecopyonread_variable_24^Read_43/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_86IdentityRead_43/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_44/DisableCopyOnReadDisableCopyOnRead%read_44_disablecopyonread_variable_23*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp%read_44_disablecopyonread_variable_23^Read_44/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_88IdentityRead_44/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_45/DisableCopyOnReadDisableCopyOnRead%read_45_disablecopyonread_variable_22*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp%read_45_disablecopyonread_variable_22^Read_45/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_90IdentityRead_45/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_46/DisableCopyOnReadDisableCopyOnRead%read_46_disablecopyonread_variable_21*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp%read_46_disablecopyonread_variable_21^Read_46/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_92IdentityRead_46/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_47/DisableCopyOnReadDisableCopyOnRead%read_47_disablecopyonread_variable_20*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp%read_47_disablecopyonread_variable_20^Read_47/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_94IdentityRead_47/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_48/DisableCopyOnReadDisableCopyOnRead%read_48_disablecopyonread_variable_19*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp%read_48_disablecopyonread_variable_19^Read_48/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_96IdentityRead_48/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_49/DisableCopyOnReadDisableCopyOnRead%read_49_disablecopyonread_variable_18*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp%read_49_disablecopyonread_variable_18^Read_49/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_98IdentityRead_49/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_50/DisableCopyOnReadDisableCopyOnRead%read_50_disablecopyonread_variable_17*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp%read_50_disablecopyonread_variable_17^Read_50/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0c
Identity_100IdentityRead_50/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_51/DisableCopyOnReadDisableCopyOnRead%read_51_disablecopyonread_variable_16*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp%read_51_disablecopyonread_variable_16^Read_51/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0c
Identity_102IdentityRead_51/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_52/DisableCopyOnReadDisableCopyOnRead%read_52_disablecopyonread_variable_15*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp%read_52_disablecopyonread_variable_15^Read_52/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_104IdentityRead_52/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_53/DisableCopyOnReadDisableCopyOnRead%read_53_disablecopyonread_variable_14*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp%read_53_disablecopyonread_variable_14^Read_53/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_106IdentityRead_53/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_54/DisableCopyOnReadDisableCopyOnRead%read_54_disablecopyonread_variable_13*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp%read_54_disablecopyonread_variable_13^Read_54/DisableCopyOnRead*
_output_shapes
:	�*
dtype0b
Identity_108IdentityRead_54/ReadVariableOp:value:0*
T0*
_output_shapes
:	�h
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:	�k
Read_55/DisableCopyOnReadDisableCopyOnRead%read_55_disablecopyonread_variable_12*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp%read_55_disablecopyonread_variable_12^Read_55/DisableCopyOnRead*
_output_shapes
:	�*
dtype0b
Identity_110IdentityRead_55/ReadVariableOp:value:0*
T0*
_output_shapes
:	�h
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:	�k
Read_56/DisableCopyOnReadDisableCopyOnRead%read_56_disablecopyonread_variable_11*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp%read_56_disablecopyonread_variable_11^Read_56/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_112IdentityRead_56/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
:k
Read_57/DisableCopyOnReadDisableCopyOnRead%read_57_disablecopyonread_variable_10*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp%read_57_disablecopyonread_variable_10^Read_57/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_114IdentityRead_57/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
:j
Read_58/DisableCopyOnReadDisableCopyOnRead$read_58_disablecopyonread_variable_9*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp$read_58_disablecopyonread_variable_9^Read_58/DisableCopyOnRead*
_output_shapes
:	d�*
dtype0b
Identity_116IdentityRead_58/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�h
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
:	d�j
Read_59/DisableCopyOnReadDisableCopyOnRead$read_59_disablecopyonread_variable_8*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp$read_59_disablecopyonread_variable_8^Read_59/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_118IdentityRead_59/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_60/DisableCopyOnReadDisableCopyOnRead$read_60_disablecopyonread_variable_7*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp$read_60_disablecopyonread_variable_7^Read_60/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0c
Identity_120IdentityRead_60/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��j
Read_61/DisableCopyOnReadDisableCopyOnRead$read_61_disablecopyonread_variable_6*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp$read_61_disablecopyonread_variable_6^Read_61/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_122IdentityRead_61/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_62/DisableCopyOnReadDisableCopyOnRead$read_62_disablecopyonread_variable_5*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp$read_62_disablecopyonread_variable_5^Read_62/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0c
Identity_124IdentityRead_62/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��j
Read_63/DisableCopyOnReadDisableCopyOnRead$read_63_disablecopyonread_variable_4*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp$read_63_disablecopyonread_variable_4^Read_63/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_126IdentityRead_63/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_64/DisableCopyOnReadDisableCopyOnRead$read_64_disablecopyonread_variable_3*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp$read_64_disablecopyonread_variable_3^Read_64/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0c
Identity_128IdentityRead_64/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��j
Read_65/DisableCopyOnReadDisableCopyOnRead$read_65_disablecopyonread_variable_2*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp$read_65_disablecopyonread_variable_2^Read_65/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_130IdentityRead_65/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_66/DisableCopyOnReadDisableCopyOnRead$read_66_disablecopyonread_variable_1*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp$read_66_disablecopyonread_variable_1^Read_66/DisableCopyOnRead*
_output_shapes
:	�*
dtype0b
Identity_132IdentityRead_66/ReadVariableOp:value:0*
T0*
_output_shapes
:	�h
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Read_67/DisableCopyOnReadDisableCopyOnRead"read_67_disablecopyonread_variable*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp"read_67_disablecopyonread_variable^Read_67/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_134IdentityRead_67/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
:L

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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*�
value�B�EB/optimizer/iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0_operations/1/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/1/bias/.ATTRIBUTES/VARIABLE_VALUEB0_operations/2/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/2/bias/.ATTRIBUTES/VARIABLE_VALUEB0_operations/3/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/3/bias/.ATTRIBUTES/VARIABLE_VALUEB0_operations/4/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/4/bias/.ATTRIBUTES/VARIABLE_VALUEB0_operations/5/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/5/bias/.ATTRIBUTES/VARIABLE_VALUEB0_operations/6/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/6/bias/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB<optimizer/_trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*�
value�B�EB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *S
dtypesI
G2E	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_136Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_137IdentityIdentity_136:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_137Identity_137:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=E9

_output_shapes
: 

_user_specified_nameConst:(D$
"
_user_specified_name
Variable:*C&
$
_user_specified_name
Variable_1:*B&
$
_user_specified_name
Variable_2:*A&
$
_user_specified_name
Variable_3:*@&
$
_user_specified_name
Variable_4:*?&
$
_user_specified_name
Variable_5:*>&
$
_user_specified_name
Variable_6:*=&
$
_user_specified_name
Variable_7:*<&
$
_user_specified_name
Variable_8:*;&
$
_user_specified_name
Variable_9:+:'
%
_user_specified_nameVariable_10:+9'
%
_user_specified_nameVariable_11:+8'
%
_user_specified_nameVariable_12:+7'
%
_user_specified_nameVariable_13:+6'
%
_user_specified_nameVariable_14:+5'
%
_user_specified_nameVariable_15:+4'
%
_user_specified_nameVariable_16:+3'
%
_user_specified_nameVariable_17:+2'
%
_user_specified_nameVariable_18:+1'
%
_user_specified_nameVariable_19:+0'
%
_user_specified_nameVariable_20:+/'
%
_user_specified_nameVariable_21:+.'
%
_user_specified_nameVariable_22:+-'
%
_user_specified_nameVariable_23:+,'
%
_user_specified_nameVariable_24:++'
%
_user_specified_nameVariable_25:+*'
%
_user_specified_nameVariable_26:+)'
%
_user_specified_nameVariable_27:+('
%
_user_specified_nameVariable_28:+''
%
_user_specified_nameVariable_29:+&'
%
_user_specified_nameVariable_30:+%'
%
_user_specified_nameVariable_31:+$'
%
_user_specified_nameVariable_32:+#'
%
_user_specified_nameVariable_33:+"'
%
_user_specified_nameVariable_34:+!'
%
_user_specified_nameVariable_35:+ '
%
_user_specified_nameVariable_36:+'
%
_user_specified_nameVariable_37:+'
%
_user_specified_nameVariable_38:+'
%
_user_specified_nameVariable_39:+'
%
_user_specified_nameVariable_40:+'
%
_user_specified_nameVariable_41:+'
%
_user_specified_nameVariable_42:+'
%
_user_specified_nameVariable_43:+'
%
_user_specified_nameVariable_44:+'
%
_user_specified_nameVariable_45:+'
%
_user_specified_nameVariable_46:+'
%
_user_specified_nameVariable_47:+'
%
_user_specified_nameVariable_48:+'
%
_user_specified_nameVariable_49:+'
%
_user_specified_nameVariable_50:+'
%
_user_specified_nameVariable_51:+'
%
_user_specified_nameVariable_52:+'
%
_user_specified_nameVariable_53:+'
%
_user_specified_nameVariable_54:+'
%
_user_specified_nameVariable_55:+'
%
_user_specified_nameVariable_56:+'
%
_user_specified_nameVariable_57:+
'
%
_user_specified_nameVariable_58:+	'
%
_user_specified_nameVariable_59:+'
%
_user_specified_nameVariable_60:+'
%
_user_specified_nameVariable_61:+'
%
_user_specified_nameVariable_62:+'
%
_user_specified_nameVariable_63:+'
%
_user_specified_nameVariable_64:+'
%
_user_specified_nameVariable_65:+'
%
_user_specified_nameVariable_66:+'
%
_user_specified_nameVariable_67:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
inputs/
serving_default_inputs:0���������<
output_00
StatefulPartitionedCall:0���������tensorflow/serving/predict:�T
�
_tracked
_inbound_nodes
_outbound_nodes
_losses
_operations
_layers
_build_shapes_dict
output_names
		optimizer

_default_save_signature

signatures"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�

_variables
_trainable_variables
 _trainable_variables_indices

iterations
_learning_rate

_momentums
_velocities"
_generic_user_object
�
trace_02�
"__inference_serving_default_650490�
���
FullArgSpec
args�

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
annotations� *�
����������ztrace_0
,
serving_default"
signature_map
c
_inbound_nodes
_outbound_nodes
 _losses
!	_loss_ids"
_generic_user_object
�
"_kernel
#bias
$_inbound_nodes
%_outbound_nodes
&_losses
'	_loss_ids
(_build_shapes_dict"
_generic_user_object
�
)_kernel
*bias
+_inbound_nodes
,_outbound_nodes
-_losses
.	_loss_ids
/_build_shapes_dict"
_generic_user_object
�
0_kernel
1bias
2_inbound_nodes
3_outbound_nodes
4_losses
5	_loss_ids
6_build_shapes_dict"
_generic_user_object
�
7_kernel
8bias
9_inbound_nodes
:_outbound_nodes
;_losses
<	_loss_ids
=_build_shapes_dict"
_generic_user_object
�
>_kernel
?bias
@_inbound_nodes
A_outbound_nodes
B_losses
C	_loss_ids
D_build_shapes_dict"
_generic_user_object
�
E_kernel
Fbias
G_inbound_nodes
H_outbound_nodes
I_losses
J	_loss_ids
K_build_shapes_dict"
_generic_user_object
�
L_inbound_nodes
M_outbound_nodes
N_losses
O	_loss_ids
P	arguments
Q_build_shapes_dict"
_generic_user_object
�
R_tracked
S_inbound_nodes
T_outbound_nodes
U_losses
V_operations
W_layers
X_build_shapes_dict
Youtput_names
Z_default_save_signature"
_generic_user_object
�
0
1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15
i16
j17
k18
l19
m20
n21
o22
p23
q24
r25
s26
t27
u28
v29
w30
x31
y32
z33
{34
|35
}36
~37
38
�39
�40
�41
�42
�43
�44
�45"
trackable_list_wrapper
�
"0
#1
)2
*3
04
15
76
87
>8
?9
E10
F11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21"
trackable_list_wrapper
 "
trackable_dict_wrapper
:	 2adam/iteration
: 2adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�B�
"__inference_serving_default_650490inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
4__inference_signature_wrapper_serving_default_650540inputs"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�

jinputs
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
:	�2dense/kernel
:�2
dense/bias
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
": 
��2dense_1/kernel
:�2dense_1/bias
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
": 
��2dense_2/kernel
:�2dense_2/bias
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
": 
��2dense_3/kernel
:�2dense_3/bias
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
!:	�d2dense_4/kernel
:d2dense_4/bias
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
!:	�d2dense_5/kernel
:d2dense_5/bias
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�
�trace_02�
"__inference_serving_default_650626�
���
FullArgSpec
args�

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
annotations� *�
����������dz�trace_0
+:)	�2adam/dense_kernel_momentum
+:)	�2adam/dense_kernel_velocity
%:#�2adam/dense_bias_momentum
%:#�2adam/dense_bias_velocity
.:,
��2adam/dense_1_kernel_momentum
.:,
��2adam/dense_1_kernel_velocity
':%�2adam/dense_1_bias_momentum
':%�2adam/dense_1_bias_velocity
.:,
��2adam/dense_2_kernel_momentum
.:,
��2adam/dense_2_kernel_velocity
':%�2adam/dense_2_bias_momentum
':%�2adam/dense_2_bias_velocity
.:,
��2adam/dense_3_kernel_momentum
.:,
��2adam/dense_3_kernel_velocity
':%�2adam/dense_3_bias_momentum
':%�2adam/dense_3_bias_velocity
-:+	�d2adam/dense_4_kernel_momentum
-:+	�d2adam/dense_4_kernel_velocity
&:$d2adam/dense_4_bias_momentum
&:$d2adam/dense_4_bias_velocity
-:+	�d2adam/dense_5_kernel_momentum
-:+	�d2adam/dense_5_kernel_velocity
&:$d2adam/dense_5_bias_momentum
&:$d2adam/dense_5_bias_velocity
-:+	d�2adam/dense_6_kernel_momentum
-:+	d�2adam/dense_6_kernel_velocity
':%�2adam/dense_6_bias_momentum
':%�2adam/dense_6_bias_velocity
.:,
��2adam/dense_7_kernel_momentum
.:,
��2adam/dense_7_kernel_velocity
':%�2adam/dense_7_bias_momentum
':%�2adam/dense_7_bias_velocity
.:,
��2adam/dense_8_kernel_momentum
.:,
��2adam/dense_8_kernel_velocity
':%�2adam/dense_8_bias_momentum
':%�2adam/dense_8_bias_velocity
.:,
��2adam/dense_9_kernel_momentum
.:,
��2adam/dense_9_kernel_velocity
':%�2adam/dense_9_bias_momentum
':%�2adam/dense_9_bias_velocity
.:,	�2adam/dense_10_kernel_momentum
.:,	�2adam/dense_10_kernel_velocity
':%2adam/dense_10_bias_momentum
':%2adam/dense_10_bias_velocity
!:	d�2dense_6/kernel
:�2dense_6/bias
": 
��2dense_7/kernel
:�2dense_7/bias
": 
��2dense_8/kernel
:�2dense_8/bias
": 
��2dense_9/kernel
:�2dense_9/bias
": 	�2dense_10/kernel
:2dense_10/bias
g
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids"
_generic_user_object
�
�_kernel
	�bias
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids
�_build_shapes_dict"
_generic_user_object
�
�_kernel
	�bias
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids
�_build_shapes_dict"
_generic_user_object
�
�_kernel
	�bias
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids
�_build_shapes_dict"
_generic_user_object
�
�_kernel
	�bias
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids
�_build_shapes_dict"
_generic_user_object
�
�_kernel
	�bias
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids
�_build_shapes_dict"
_generic_user_object
�B�
"__inference_serving_default_650626inputs"�
���
FullArgSpec
args�

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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper�
"__inference_serving_default_650490v "#)*0178>?EF����������/�,
%�"
 �
inputs���������
� "!�
unknown����������
"__inference_serving_default_650626j����������/�,
%�"
 �
inputs���������d
� "!�
unknown����������
4__inference_signature_wrapper_serving_default_650540� "#)*0178>?EF����������9�6
� 
/�,
*
inputs �
inputs���������"3�0
.
output_0"�
output_0���������