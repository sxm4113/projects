`timescale 1ns/1ps
module combi_test_0(
    input [7:0] i_value_a,
    input [7:0] i_value_b,
    output [7:0] o_value_add,
    output [7:0] o_value_sub,
    output [7:0] o_value_multi
);
    assign o_value_add = i_value_a + i_value_b;
    assign o_value_sub = i_value_a - i_value_b;
    assign o_value_multi = i_value_a * i_value_b;

endmodule
