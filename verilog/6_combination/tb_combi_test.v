`timescale 1ns / 1ps
module tb_combi_test;
    reg clk;
    reg  [7:0] i_value_a;
    reg  [7:0] i_value_b;
    wire [7:0] o_value_add_0;
    wire [7:0] o_value_sub_0;
    wire [7:0] o_value_multi_0;
    wire [7:0] o_value_add_1;
    wire [7:0] o_value_sub_1;
    wire [7:0] o_value_multi_1;
    
    always #5 clk = ~clk;

    initial begin
        clk <= 0;
        #20
            i_value_a = 8'd10;
            i_value_b = 8'd2;

        #20
    $finish;
    end
    combi_test_0 dut_0 (
        .i_value_a(i_value_a),
        .i_value_b(i_value_b),
        .o_value_add(o_value_add_0),
        .o_value_sub(o_value_sub_0),
        .o_value_multi(o_value_multi_0)
    );

    combi_test_1 dut_1 (
        .i_value_a(i_value_a),
        .i_value_b(i_value_b),
        .o_value_add(o_value_add_1),
        .o_value_sub(o_value_sub_1),
        .o_value_multi(o_value_multi_1)
    );

endmodule