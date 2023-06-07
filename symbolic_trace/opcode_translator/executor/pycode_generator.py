# This class is used for abstract code generation:
# We only need to care about what type of bytecode our code needs to generate,
# without worrying about the subscripts of bytecode instructions in the code option.

from __future__ import annotations

import dis
import types

import opcode

from ...utils import (
    ResumeFnNameFactory,
    list_contain_by_id,
    list_find_index_by_id,
)
from ..instruction_utils import (
    gen_instr,
    get_instructions,
    instrs_info,
    modify_instrs,
    modify_vars,
)
from ..instruction_utils.opcode_analysis import analysis_inputs

'''
    code options for PyCodeObject
'''

pycode_attributes = [
    "co_argcount",
    "co_posonlyargcount",
    "co_kwonlyargcount",
    "co_nlocals",
    "co_stacksize",
    "co_flags",
    "co_code",
    "co_consts",
    "co_names",
    "co_varnames",
    "co_filename",
    "co_name",
    "co_firstlineno",
    "co_lnotab",
    "co_freevars",
    "co_cellvars",
]


def gen_code_options(code):
    code_options = {}
    for k in pycode_attributes:
        val = getattr(code, k)
        if isinstance(val, tuple):
            val = list(val)
        code_options[k] = val
    return code_options


'''
    generator a new code object
'''


def gen_new_opcode(instrs, code_options, keys):
    bytecode, lnotab = assemble(instrs, code_options["co_firstlineno"])
    code_options["co_lnotab"] = lnotab
    code_options["co_code"] = bytecode
    code_options["co_nlocals"] = len(code_options["co_varnames"])
    code_options["co_stacksize"] = stacksize(instrs)
    for key, val in code_options.items():
        if isinstance(val, list):
            code_options[key] = tuple(val)
    # code_options is a dict, use keys to makesure the input order
    return types.CodeType(*[code_options[k] for k in keys])


# list of instructions => bytecode & lnotab
def assemble(instructions, firstlineno):
    cur_line = firstlineno
    cur_bytecode = 0

    code = []
    lnotab = []

    for instr in instructions:
        # set lnotab
        if instr.starts_line is not None:
            line_offset = instr.starts_line - cur_line
            bytecode_offset = len(code) - cur_bytecode

            cur_line = instr.starts_line
            cur_bytecode = len(code)

            lnotab.extend(modify_lnotab(bytecode_offset, line_offset))

        # get bytecode
        arg = instr.arg or 0
        code.extend((instr.opcode, arg & 0xFF))

    return bytes(code), bytes(lnotab)


def to_byte(num):
    if num < 0:
        num += 256  #  -1 => 255
    return num


def modify_lnotab(byte_offset, line_offset):
    if byte_offset > 127:
        ret = []
        while byte_offset > 127:
            ret.extend((127, 0))
            byte_offset -= 127
        # line_offset might > 127, call recursively
        ret.extend(modify_lnotab(byte_offset, line_offset))
        return ret

    if line_offset > 127:
        # here byte_offset < 127
        ret = [byte_offset, 127]
        line_offset -= 127
        while line_offset > 0:
            ret.extend((0, line_offset))
            line_offset -= 127
        return ret

    # both < 127
    return [to_byte(byte_offset), to_byte(line_offset)]


# TODO: need to update
def stacksize(instructions):
    # two list below shows the possible stack size before opcode is called
    # the stack size might be different in different branch, so it has max and min
    max_stack = [float("-inf")] * len(instructions)
    min_stack = [float("inf")] * len(instructions)

    max_stack[0] = 0
    min_stack[0] = 0

    def update_stacksize(lasti, nexti, stack_effect):
        max_stack[nexti] = max(
            max_stack[nexti], max_stack[lasti] + stack_effect
        )
        min_stack[nexti] = min(
            min_stack[nexti], max_stack[lasti] + stack_effect
        )

    for idx in range(len(instructions)):
        instr = instructions[idx]

        if idx + 1 < len(instructions):
            stack_effect = dis.stack_effect(instr.opcode, instr.arg, jump=False)
            update_stacksize(idx, idx + 1, stack_effect)

        if instr.opcode in opcode.hasjabs or instr.opcode in opcode.hasjrel:
            stack_effect = dis.stack_effect(instr.opcode, instr.arg, jump=True)
            target_idx = instructions.index(instr.jump_to)
            update_stacksize(idx, target_idx, stack_effect)

    assert min(min_stack) >= 0
    return max(max_stack)


'''
    helper to create new code object
'''


class PyCodeGen:
    def __init__(self, frame):
        self._frame = frame
        self._origin_code = frame.f_code
        self._code_options = gen_code_options(self._origin_code)
        self._f_globals = frame.f_globals
        self._instructions = []
        self.objname_map = {}  # map from name to LOAD_GLOBAL index

    def gen_pycode(self):
        """
        return a new pycode, which is runnable.
        """
        modify_instrs(self._instructions)
        modify_vars(self._instructions, self._code_options)
        new_code = gen_new_opcode(
            self._instructions, self._code_options, pycode_attributes
        )
        return new_code

    def gen_resume_fn_at(self, index, stack_size=0):
        self._instructions = get_instructions(self._origin_code)
        # TODO(dev): could give an example code here?
        if self._instructions[index].opname == 'RETURN_VALUE':
            return None, set()
        inputs = analysis_inputs(self._instructions, index)
        self._instructions = (
            [
                gen_instr('LOAD_FAST', argval=f'__stack_arg{i}')
                for i in range(stack_size)
            ]
            + [gen_instr('JUMP_ABSOLUTE', jump_to=self._instructions[index])]
            + self._instructions
        )

        self._code_options['co_argcount'] = len(inputs) + stack_size
        # inputs should be at the front of the co_varnames
        self._code_options['co_varnames'] = tuple(
            [f'__stack_arg{i}' for i in range(stack_size)]
            + list(inputs)
            + [
                var_name
                for var_name in self._origin_code.co_varnames
                if var_name not in inputs
            ]
        )
        fn_name = ResumeFnNameFactory().next()
        self._code_options['co_name'] = fn_name

        new_code = self.gen_pycode()
        fn = types.FunctionType(new_code, self._f_globals, fn_name)

        return fn, inputs

    def _gen_fn(self, inputs):
        # outputs is same as inputs, and they are always in locals
        for name in inputs:
            self.gen_load_fast(name)

        self.gen_build_tuple(len(inputs))
        self.gen_return()

        self._code_options['co_argcount'] = len(inputs)
        self._code_options['co_varnames'] = tuple(
            list(inputs)
            + [
                var_name
                for var_name in self._origin_code.co_varnames
                if var_name not in inputs
            ]
        )
        fn_name = ResumeFnNameFactory().next()
        self._code_options['co_name'] = fn_name
        new_code = self.gen_pycode()
        fn = types.FunctionType(new_code, self._f_globals, fn_name)
        return fn

    def gen_loop_body_between(self, for_iter, start, end):
        break_flag_name = "_break_flag"
        origin_instrs = get_instructions(self._origin_code)
        inputs = list(analysis_inputs(origin_instrs, start)) + [break_flag_name]

        # for balance the stack (the loop body will pop iter first before break or return)
        # this None is used for replace the iterator obj in stack top
        self.gen_load_const(None)

        # extend loop body main logic
        self.extend_instrs(origin_instrs[start:end])

        # break should jump to this nop
        nop_for_break = self._add_instr("NOP")

        # need do additional operates when break
        self.gen_load_const(False)
        self.gen_store_fast(break_flag_name)
        self.gen_load_const(None)  # keep stack balance

        # continue should jump to this nop
        nop_for_continue = self._add_instr("NOP")
        self.gen_pop_top()

        out_loop = for_iter.jump_to
        for instr in self._instructions:
            if instr.jump_to == for_iter:
                instr.jump_to = nop_for_continue
            if instr.jump_to == out_loop:
                instr.jump_to = nop_for_break

        return self._gen_fn(inputs), inputs

    def gen_for_loop_fn_between(self, iterator, start, end):
        origin_instrs = get_instructions(self._origin_code)
        inputs = list(analysis_inputs(origin_instrs, start)) + [iterator.id]
        self.gen_load_fast(iterator.id)
        self.extend_instrs(origin_instrs[start:end])
        for_iter = origin_instrs[start]
        out_loop_instr = origin_instrs[start].jump_to

        nop_for_continue = self._add_instr("NOP")
        jump = self._add_instr("JUMP_ABSOLUTE", jump_to=for_iter)
        nop_for_break = self._add_instr("NOP")

        for instr in self._instructions:
            if instr.jump_to == for_iter:
                instr.jump_to = nop_for_continue

            if instr.jump_to == out_loop_instr:
                instr.jump_to = nop_for_break

        jump.jump_to = for_iter

        return self._gen_fn(inputs), inputs

    def gen_load_closure(self, name):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("LOAD_CLOSURE", arg=idx, argval=name)

    def gen_load_const(self, value):
        # Python `list.index` will find an item equal to query, i.e. `query == item`
        # returns a value of True. Since `1 == True`, this will result in an incorrect
        # index. To avoid this problem, we use id for comparison.
        if not list_contain_by_id(self._code_options["co_consts"], value):
            self._code_options["co_consts"].append(value)
        idx = list_find_index_by_id(self._code_options["co_consts"], value)
        self._add_instr("LOAD_CONST", arg=idx, argval=value)

    def gen_load_global(self, name):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("LOAD_GLOBAL", arg=idx, argval=name)

    def gen_load_object(self, obj, obj_name):
        if obj_name not in self.objname_map:
            self._f_globals[obj_name] = obj
            self._code_options["co_names"].append(obj_name)
            idx = len(self._code_options["co_names"]) - 1
            self.objname_map[obj_name] = idx
        idx = self.objname_map[obj_name]
        self._add_instr("LOAD_GLOBAL", arg=idx, argval=obj_name)

    def gen_load_fast(self, name):
        if name not in self._code_options["co_varnames"]:
            self._code_options["co_varnames"].append(name)
        idx = self._code_options["co_varnames"].index(name)
        self._add_instr("LOAD_FAST", arg=idx, argval=name)

    def gen_load_attr(self, name: str):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("LOAD_ATTR", arg=idx, argval=name)

    def gen_load_method(self, name: str):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("LOAD_METHOD", arg=idx, argval=name)

    def gen_import_name(self, name: str):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("IMPORT_NAME", arg=idx, argval=name)

    def gen_push_null(self):
        # There is no PUSH_NULL bytecode before python3.11, so we push
        # a NULL element to the stack through the following bytecode
        self.gen_load_const(0)
        self.gen_load_const(None)
        self.gen_import_name('sys')
        self.gen_store_fast('sys')
        self.gen_load_fast('sys')
        self.gen_load_method('getsizeof')
        self._add_instr("POP_TOP")
        # TODO(dev): push NULL element to the stack through PUSH_NULL bytecode in python3.11

    def gen_store_fast(self, name):
        if name not in self._code_options["co_varnames"]:
            self._code_options["co_varnames"].append(name)
        idx = self._code_options["co_varnames"].index(name)
        self._add_instr("STORE_FAST", arg=idx, argval=name)

    def gen_subscribe(self):
        self._add_instr("BINARY_SUBSCR")

    def gen_build_tuple(self, count):
        self._add_instr("BUILD_TUPLE", arg=count, argval=count)

    def gen_build_list(self, count):
        self._add_instr("BUILD_LIST", arg=count, argval=count)

    def gen_build_map(self, count):
        self._add_instr("BUILD_MAP", arg=count, argval=count)

    def gen_unpack_sequence(self, count):
        self._add_instr("UNPACK_SEQUENCE", arg=count, argval=count)

    def gen_call_function(self, argc=0):
        self._add_instr("CALL_FUNCTION", arg=argc, argval=argc)

    def gen_pop_top(self):
        self._add_instr("POP_TOP")

    def gen_return(self):
        self._add_instr("RETURN_VALUE")

    def add_pure_instructions(self, instructions):
        """
        add instructions and do nothing.
        """
        self._instructions.extend(instructions)

    def _add_instr(self, *args, **kwargs):
        instr = gen_instr(*args, **kwargs)
        self._instructions.append(instr)
        return instr

    def _insert_instr(self, index, *args, **kwargs):
        instr = gen_instr(*args, **kwargs)
        self._instructions.insert(index, instr)

    def pprint(self):
        print(instrs_info(self._instructions))

    def extend_instrs(self, instrs):
        self._instructions.extend(instrs)

    def pop_instr(self):
        self._instructions.pop()
