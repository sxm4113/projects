`timescale 1ns/1ps

module combi_test_1(
    input [7:0] i_value_a,
    input [7:0] i_value_b,

    output [7:0] o_value_add,
    output [7:0] o_value_sub,
    output [7:0] o_value_multi
);
    reg [7:0] r_value_add;
    reg [7:0] r_value_sub;
    reg [7:0] r_value_multi;

    always@(*) begin
        r_value_add = i_value_a + i_value_b;
        r_value_sub = i_value_a - i_value_b;
        r_value_multi = i_value_a * i_value_b;
    end

    assign o_value_add = r_value_add;
    assign o_value_sub = r_value_sub;
    assign o_value_multi = r_value_multi;

endmodule